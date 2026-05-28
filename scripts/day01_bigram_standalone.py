"""
Day 1 — Build an LLM from scratch
==================================
Goal: data -> tokens -> simplest possible language model (bigram) -> train -> sample.

This is deliberately the *minimal* model. The point isn't that it's good (it
won't be), it's that the training loop, the batching, and the encode/decode
plumbing here are EXACTLY what we'll reuse for the real transformer. Every later
day just swaps out the `model` and keeps this skeleton.

A "bigram" model predicts the next character using only the current character —
a lookup table of shape (vocab, vocab). No context, no attention, no notion of
position. That limitation is the whole motivation for self-attention on Day 2.

Run:  python day01_bigram.py
Needs: pip install torch    (CPU is fine for today)
"""

import os
import urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters. Small on purpose — this should train in seconds on CPU.
# ---------------------------------------------------------------------------
BATCH_SIZE = 32          # how many independent sequences we process in parallel
BLOCK_SIZE = 8           # max context length (irrelevant for bigram, used Day 2+)
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 200         # how many batches we average over to estimate loss
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1337

torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Data. Tiny Shakespeare (~1MB of text). Small enough to overfit, big enough
#    to be interesting. Downloads once, then caches locally.
# ---------------------------------------------------------------------------
DATA_PATH = "input.txt"
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt")

if not os.path.exists(DATA_PATH):
    print("Downloading tiny shakespeare...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset length in characters: {len(text):,}")

# ---------------------------------------------------------------------------
# 2. Tokenization (the simplest kind: character-level).
#    Build a vocab of every unique char, map char<->int. Real models use
#    subword BPE instead — that's Day 7. The idea is identical: text <-> ints.
# ---------------------------------------------------------------------------
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int -> string
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# Encode the whole dataset into one long tensor of token ids.
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split — never measure progress on data you trained on.
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Grab BATCH_SIZE random chunks of length BLOCK_SIZE.
    x is the input, y is x shifted one position right (the next-token target)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    """Average loss over EVAL_ITERS batches for train and val. Less noisy than
    looking at a single step's loss."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# 3. The model. A single embedding table of shape (vocab_size, vocab_size).
#    Row i holds the unnormalized log-probabilities (logits) for what token
#    follows token i. That's the entire model. No context beyond one token.
# ---------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) integers. Look up -> logits: (B, T, vocab_size).
        logits = self.token_embedding_table(idx)

        if targets is None:
            return logits, None

        # Cross-entropy wants (N, C) logits and (N,) targets, so flatten the
        # batch & time dims together.
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive sampling: predict next token, append, repeat."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)              # (B, T, C)
            logits = logits[:, -1, :]          # only the last step matters
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # sample, not argmax
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# 4. Train. AdamW is the optimizer used (in fancier form) all the way up to
#    frontier models. This loop is unchanged for the rest of the course.
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for it in range(MAX_ITERS):
    if it % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(f"step {it:5d} | train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------------------------------------------------------
# 5. Sample. Start from a single token (newline = id 0) and let it ramble.
# ---------------------------------------------------------------------------
print("\n--- sample ---")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, max_new_tokens=400)[0].tolist()))

# ---------------------------------------------------------------------------
# WHAT TO EXPECT
#   Loss starts near ln(vocab_size) ~= 4.17 (random guessing) and drops to
#   ~2.4-2.5. The sample looks like Shakespeare-flavored gibberish: correct
#   character frequencies, plausible word lengths, but no real words. That's
#   the ceiling of a model that can only see ONE character of context.
#
# THE GAP TO FRONTIER
#   Everything: context (1 token vs 200k+), parameters (~4k vs ~10^11-10^12),
#   tokenization (chars vs BPE), data (1MB vs ~10^13 tokens). Day 2 attacks the
#   single biggest one first — giving the model the ability to look back at
#   previous tokens via self-attention.
# ---------------------------------------------------------------------------
