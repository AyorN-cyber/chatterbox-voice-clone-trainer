# chatterbox-voice-trainer

A small, readable template for fine-tuning the **T3** text-to-speech-token model of
[Chatterbox TTS](https://github.com/resemble-ai/chatterbox) on **your own voice**, on a free Kaggle (or any CUDA) GPU.
Clone it, drop in your clips, run two commands, listen to the results.

> Not affiliated with Resemble AI. Only train on a voice you own or have clear permission to use.

## What it does

1. Reads a folder of short clips plus a `metadata.csv` (`filename|transcript` per line).
2. Pre-computes speech tokens (S3 tokenizer), a speaker embedding (voice encoder) and text tokens once, so training is cheap.
3. Freezes everything except T3 and trains it with teacher-forced cross-entropy on the speech tokens.
   Gradient checkpointing, mixed precision and gradient accumulation keep memory low.
4. Holds a few clips out, reports held-out loss before training and after every epoch, and saves a checkpoint per epoch.
5. `generate_samples.py` renders the same sentences with the base model and every checkpoint so you can compare by ear.

## Quick start

```bash
pip install -r requirements.txt
mkdir -p data && cp /path/to/your/clips/* data/ && cp metadata.example.csv data/metadata.csv   # then edit it
EPOCHS=6 LR=1e-5 python train_t3.py data runs/my_voice
python generate_samples.py runs/my_voice data/clip_001.wav "Hello, this is a test." "Another sentence to compare."
```

Kaggle: add your clips as a **private** dataset, enable a GPU, `!pip -q install chatterbox-tts`, then run the same commands with the dataset path from `/kaggle/input`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `EPOCHS` | 6 | passes over the training clips |
| `LR` | 1e-5 | learning rate |
| `MAXCLIPS` | 0 | cap on clips (0 = all), handy for a quick smoke test |
| `HELD_OUT` | 6 | clips kept out of training for the held-out loss |
| `ACCUM` | 4 | gradient accumulation steps |
| `OPT` | adamw | `adamw` or `sgd` (sgd uses less memory) |
| `SMALL` | unset | set it to train only the speech head and the last two layers, for small GPUs |

## Practical notes

- Start with a smoke test (`MAXCLIPS=10 EPOCHS=1`) before spending GPU hours.
- Clean audio and exact transcripts matter more than quantity. Remove clipped, noisy or mis-transcribed clips.
- Held-out loss that stops falling while train loss keeps dropping means over-fitting. Prefer an earlier checkpoint and trust your ears.
- Checkpoints are fp16 `safetensors` of the T3 weights only. Load one into `tts.t3` as `generate_samples.py` does.
- Chatterbox embeds an imperceptible watermark in generated audio. Keep it, and label synthetic voices as synthetic.

## Files

`train_t3.py` training, `generate_samples.py` listening test, `metadata.example.csv` data format, `requirements.txt` dependencies.

## License

MIT for this repo. Chatterbox itself is MIT-licensed by Resemble AI.
