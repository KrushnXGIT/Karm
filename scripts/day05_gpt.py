"""Day 5 — train the full GPT end-to-end on tiny-shakespeare.

Reuses the same train_model loop from Day 1 (the model is swappable; the
skeleton is unchanged). On CPU this takes a few minutes; if you have a CUDA
GPU it's used automatically and finishes far faster.

Run from the repo root:  python scripts/day05_gpt.py
"""
import yaml

from karm.data import CharTokenizer, load_text, make_splits
from karm.models.gpt import GPT
from karm.sample import generate_text
from karm.train import train_model
from karm.utils import count_params, get_device, set_seed

cfg = yaml.safe_load(open("configs/gpt_char.yaml"))
device = get_device()
set_seed(cfg["seed"])

text = load_text()
tok = CharTokenizer(text)
train_data, val_data = make_splits(text, tok)

model = GPT(
    vocab_size=tok.vocab_size,
    n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"],
    block_size=cfg["block_size"], dropout=cfg["dropout"],
).to(device)
print(f"device={device}  vocab={tok.vocab_size}  params={count_params(model):,}")
print("training... (CPU: a few minutes; loss should fall well below the "
      "bigram's 2.45 floor)\n")

train_model(
    model, train_data, val_data,
    block_size=cfg["block_size"], batch_size=cfg["batch_size"],
    max_iters=cfg["max_iters"], eval_interval=cfg["eval_interval"],
    eval_iters=cfg["eval_iters"], learning_rate=cfg["learning_rate"],
    device=device,
)

print("\n--- sample ---")
print(generate_text(model, tok, device, max_new_tokens=500))
