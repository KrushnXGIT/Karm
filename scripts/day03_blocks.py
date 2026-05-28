"""Day 3 — Multi-head attention + feedforward.

Confirms the key property: both modules take (B, T, n_embd) and return
(B, T, n_embd). Shape in == shape out is what lets us stack them and add
residual connections on Day 4.

Run from the repo root:  python scripts/day03_blocks.py
"""
import torch

from karm.models.gpt import FeedForward, MultiHeadAttention

torch.manual_seed(1337)

B, T = 4, 8
n_embd, n_head, block_size = 32, 4, 8
x = torch.randn(B, T, n_embd)
print(f"input:            {tuple(x.shape)}")

# 4 heads, each of size 32//4 = 8, concatenated back to 32, then projected.
mha = MultiHeadAttention(n_embd, n_head, block_size)
a = mha(x)
print(f"after attention:  {tuple(a.shape)}  (communication: tokens mix)")

ffn = FeedForward(n_embd)
f = ffn(a)
print(f"after feedforward:{tuple(f.shape)}  (computation: per-token, no mixing)")

assert x.shape == a.shape == f.shape
print("\nshape preserved end-to-end -> these are stackable (Day 4 adds residuals)")

# A look at where the parameters live.
mha_p = sum(p.numel() for p in mha.parameters())
ffn_p = sum(p.numel() for p in ffn.parameters())
print(f"\nparams | attention: {mha_p:,}   feedforward: {ffn_p:,}")
print("Note the feedforward holds MORE params than attention -- the 4x hidden "
      "layer is where most of a transformer's weight (and 'thinking') sits.")
