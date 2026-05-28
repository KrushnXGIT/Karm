"""Data plumbing: tokenization, loading, train/val splits, batching.

This is the Day-1 logic factored out so every later model reuses it.
The tokenizer here is character-level; Day 7 swaps in a BPE tokenizer with
the same encode()/decode() interface, so nothing downstream has to change.
"""
import os
import urllib.request

import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def load_text(path="data/raw/input.txt", url=TINY_SHAKESPEARE_URL):
    """Load a text corpus, downloading tiny-shakespeare on first run."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading corpus -> {path}")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """The simplest possible tokenizer: one token per unique character."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def make_splits(text, tokenizer, train_frac=0.9):
    """Encode the whole corpus and split into train/val tensors."""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data, block_size, batch_size, device):
    """Sample batch_size random (x, y) chunks; y is x shifted by one."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
