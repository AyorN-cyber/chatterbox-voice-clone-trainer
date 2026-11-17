"""Fine-tune the T3 (text -> speech-token) model of Chatterbox TTS on your own voice.

Only T3 is trained. The speech tokenizer (S3) and the voice encoder stay frozen.
Data: a folder of audio clips plus a metadata.csv with lines of `filename|transcript`.

Usage:
    python train_t3.py DATA_DIR OUT_DIR
Env vars: EPOCHS (6), LR (1e-5), MAXCLIPS (0 = all), OPT (adamw|sgd),
          SMALL (set to train only the speech head and last layers, uses less memory),
          HELD_OUT (6), ACCUM (4)
"""
import glob, json, math, os, random, sys, time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from chatterbox.tts import ChatterboxTTS, punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond

DATA, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(os.environ.get("EPOCHS", 6))
LR = float(os.environ.get("LR", 1e-5))
MAXCLIPS = int(os.environ.get("MAXCLIPS", 0))
OPT = os.environ.get("OPT", "adamw")
HELD_OUT = int(os.environ.get("HELD_OUT", 6))
ACCUM = int(os.environ.get("ACCUM", 4))
SR = 16000

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
random.seed(0); torch.manual_seed(0)


def load_rows():
    meta = glob.glob(os.path.join(DATA, "**", "metadata.csv"), recursive=True)
    if not meta:
        sys.exit("No metadata.csv found under DATA_DIR (see metadata.example.csv)")
    audio = {os.path.basename(p): p for ext in ("wav", "flac", "mp3")
             for p in glob.glob(os.path.join(DATA, "**", f"*.{ext}"), recursive=True)}
    rows = []
    for line in open(meta[0], encoding="utf-8"):
        if "|" not in line:
            continue
        name, text = line.rstrip("\n").split("|", 1)
        path = audio.get(os.path.basename(name))
        if path:
            rows.append((path, text))
    return rows[:MAXCLIPS] if MAXCLIPS else rows


rows = load_rows()
print("clips", len(rows), flush=True)

tts = ChatterboxTTS.from_pretrained(dev)
t3, s3, ve, tok, hp = tts.t3, tts.s3gen, tts.ve, tts.tokenizer, tts.t3.hp

# Precompute everything that does not need gradients: speech tokens, reference prompt, speaker embedding, text tokens.
items = []
for path, text in rows:
    wav, _ = librosa.load(path, sr=SR)
    with torch.no_grad():
        speech, _ = s3.tokenizer.forward([wav], max_len=int(len(wav) / SR * 25) + 5)
        speech = torch.atleast_2d(speech)[0].cpu()
        prompt, _ = s3.tokenizer.forward([wav[:96000]], max_len=hp.speech_cond_prompt_len)
        prompt = torch.atleast_2d(prompt)[0].cpu()
        spk = torch.from_numpy(ve.embeds_from_wavs([wav], sample_rate=SR)).mean(0, keepdim=True).cpu()
    ids = tok.text_to_tokens(punc_norm(text))[0]
    ids = F.pad(F.pad(ids, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
    items.append(dict(text=ids, speech=speech, prompt=prompt, spk=spk))
print("prepared", len(items), flush=True)

random.shuffle(items)
held, train = items[:HELD_OUT], items[HELD_OUT:]

# Freeze everything except T3.
for p in s3.parameters(): p.requires_grad = False
for p in ve.parameters(): p.requires_grad = False
t3.train()
if hasattr(t3.tfmr, "gradient_checkpointing_enable"):
    t3.tfmr.gradient_checkpointing_enable()
if os.environ.get("SMALL"):  # memory-friendly: speech head + last transformer layers only
    n_layers = len(t3.tfmr.layers)
    last = tuple(f"layers.{i}." for i in range(n_layers - 2, n_layers))
    for n, p in t3.named_parameters():
        p.requires_grad = ("speech_head" in n) or any(k in n for k in last)
params = [p for p in t3.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR) if OPT == "sgd" else torch.optim.AdamW(params, lr=LR)
use_amp = dev == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)


def batch_loss(it, ref):
    """Teacher-forced cross-entropy on speech tokens. `ref` supplies the voice conditioning."""
    cond = T3Cond(speaker_emb=ref["spk"].to(dev),
                  cond_prompt_speech_tokens=ref["prompt"][None].to(dev),
                  emotion_adv=0.5 * torch.ones(1, 1, 1)).to(device=dev)
    text = it["text"][None].to(dev)
    sp_in = torch.cat([torch.tensor([hp.start_speech_token]), it["speech"]])[None].to(dev)
    sp_tg = torch.cat([it["speech"], torch.tensor([hp.stop_speech_token])])[None].to(dev)
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        out = t3.forward(t3_cond=cond, text_tokens=text,
                         text_token_lens=torch.tensor([text.size(1)], device=dev),
                         speech_tokens=sp_in,
                         speech_token_lens=torch.tensor([sp_in.size(1)], device=dev))
    return F.cross_entropy(out.speech_logits[0].float(), sp_tg[0])


def evaluate():
    t3.eval(); losses = []
    with torch.no_grad():
        for i, it in enumerate(held):
            losses.append(batch_loss(it, held[(i + 1) % len(held)]).item())
    t3.train()
    return float(np.mean(losses))


log, step = [], 0
print("eval before", evaluate(), flush=True)
for ep in range(1, EPOCHS + 1):
    random.shuffle(train); tl = []
    for i, it in enumerate(train):
        loss = batch_loss(it, random.choice(train))
        scaler.scale(loss / ACCUM).backward(); tl.append(loss.item())
        if (i + 1) % ACCUM == 0 or i == len(train) - 1:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
    ev = evaluate()
    log.append(dict(epoch=ep, train=float(np.mean(tl)), held=ev))
    print(log[-1], flush=True)
    sd = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in t3.state_dict().items()}
    save_file(sd, os.path.join(OUT, f"t3_ep{ep}.safetensors"))
json.dump(log, open(os.path.join(OUT, "log.json"), "w"), indent=2)
print("done", flush=True)
