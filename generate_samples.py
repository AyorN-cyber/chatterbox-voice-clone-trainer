"""Generate test sentences with the base model and every fine-tuned checkpoint, so you can listen and compare.

Usage:
    python generate_samples.py RUN_DIR REFERENCE_CLIP.wav "Sentence one." "Sentence two."
"""
import glob, os, sys

import torch, torchaudio
from safetensors.torch import load_file
from chatterbox.tts import ChatterboxTTS

run_dir, ref, lines = sys.argv[1], sys.argv[2], sys.argv[3:]
out_dir = os.path.join(run_dir, "samples"); os.makedirs(out_dir, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tts = ChatterboxTTS.from_pretrained(dev)
base = {k: v.clone() for k, v in tts.t3.state_dict().items()}
epochs = sorted(int(os.path.basename(p)[5:-12]) for p in glob.glob(os.path.join(run_dir, "t3_ep*.safetensors")))
for ep in [0] + epochs:  # 0 = the untouched base model
    if ep:
        tts.t3.load_state_dict({k: v.float() for k, v in load_file(os.path.join(run_dir, f"t3_ep{ep}.safetensors")).items()})
    else:
        tts.t3.load_state_dict(base)
    for i, line in enumerate(lines):
        torch.manual_seed(1)
        wav = tts.generate(line, audio_prompt_path=ref, exaggeration=0.5, cfg_weight=0.3)
        torchaudio.save(os.path.join(out_dir, f"ep{ep}_L{i}.wav"), wav.cpu(), tts.sr)
print(sorted(os.listdir(out_dir)))
