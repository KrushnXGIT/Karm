# Karm — Build an LLM from Scratch

A step-by-step project to build a real, modern transformer LLM from first
principles, then understand exactly what separates it from a frontier model.

The honest framing: the gap to an Opus-class model is scale, data, compute, and
post-training — not architecture you're missing. We build the real architecture
small, train it on real data, then map the scaling wall at every step.

## Quickstart (Windows / PowerShell)

```powershell
# from C:\Users\HP\Desktop\Karm
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .            # installs deps + makes `karm` importable
python scripts\day01_bigram.py
pytest                      # sanity checks
```

If activating the venv is blocked, run once:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Layout

```
configs/      hyperparameters per experiment (yaml)
data/         corpora (raw/ + processed/, gitignored)
checkpoints/  saved weights (gitignored)
logs/         training logs / loss curves (gitignored)
src/karm/     the reusable library  <-- new code goes here
  data.py       tokenizers, splits, batching
  models/       bigram.py, gpt.py (built up Days 2-6)
  train.py      model-agnostic training loop
  sample.py     generation
  utils.py      device, seeding, checkpoints
scripts/      thin day-by-day runners (import from karm)
tests/        pytest sanity checks
notebooks/    scratch / exploration
ai_collab/    coordination layer for working across Claude + Gemini + ChatGPT
```

## The rule of the repo
Logic lives in `src/karm/`. `scripts/dayNN_*.py` are thin runners. When a day
introduces something reusable, it goes in the library — never copy-pasted.

## Roadmap
See `ai_collab/ROADMAP.md` for the full 5-phase plan.
