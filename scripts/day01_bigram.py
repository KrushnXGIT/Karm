"""Day 1 entrypoint — the bigram baseline, now using the `karm` package.

Compare this ~20-line runner against day01_bigram_standalone.py (your original
monolith). Same result, but the data/model/train logic now lives in src/karm/
and gets reused every future day instead of being copy-pasted.

Run from the repo root:  python scripts/day01_bigram.py
(Requires `pip install -e .` first so `karm` is importable.)
"""
import yaml

from karm.data import load_text, CharTokenizer, make_splits
from karm.models import BigramLanguageModel
from karm.sample import generate_text
from karm.train import train_model
from karm.utils import get_device, set_seed, count_params

cfg = yaml.safe_load(open("configs/bigram.yaml"))
device = get_device()
set_seed(cfg["seed"])

text = load_text()
tok = CharTokenizer(text)
train_data, val_data = make_splits(text, tok)
print(f"device={device}  vocab={tok.vocab_size}  chars={len(text):,}")

model = BigramLanguageModel(tok.vocab_size).to(device)
print(f"params: {count_params(model):,}")

train_model(
    model, train_data, val_data,
    block_size=cfg["block_size"], batch_size=cfg["batch_size"],
    max_iters=cfg["max_iters"], eval_interval=cfg["eval_interval"],
    eval_iters=cfg["eval_iters"], learning_rate=cfg["learning_rate"],
    device=device,
)

print("\n--- sample ---")
print(generate_text(model, tok, device, max_new_tokens=400))
