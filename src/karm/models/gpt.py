"""The real GPT, assembled across Days 2-6.

Build order (each class added + tested in isolation before wiring together):
  Day 2  Head               - single scaled dot-product self-attention head [DONE]
  Day 3  MultiHeadAttention - parallel heads + projection
         FeedForward        - position-wise MLP
  Day 4  Block              - attn + ff with residuals & LayerNorm (pre-norm)
  Day 5  GPT                - token + positional embeddings, N blocks, lm_head
  Day 6  generate()         - sampling with temperature / top-k

Phase 3 (Days 13-18) then upgrades this to a modern Llama-style stack:
RoPE, RMSNorm, SwiGLU, KV cache, grouped-query attention.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One causal self-attention head.

    Each token projects to a query, key, and value. Attention weights are the
    scaled query-key affinities, masked so a token can only attend to itself
    and earlier tokens, then softmaxed. Output is the weighted sum of values.
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril isn't a parameter (nothing to learn) but must move with the
        # model to GPU and be saved -- that's what register_buffer is for.
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # scaled dot-product affinities; the 1/sqrt(d_k) keeps softmax soft
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        # causal mask: slice tril to the actual sequence length T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)   # (B, T, head_size)
        return wei @ v      # (B, T, head_size)
