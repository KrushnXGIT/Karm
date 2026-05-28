"""The GPT, assembled across Days 2-6.

  Day 2  Head               - single scaled dot-product self-attention head [DONE]
  Day 3  MultiHeadAttention - parallel heads + projection                   [DONE]
         FeedForward        - position-wise MLP                             [DONE]
  Day 4  LayerNorm          - from scratch, matches nn.LayerNorm            [DONE]
         Block              - pre-norm: residual MHA + residual FFN         [DONE]
  Day 5  GPT                - embeddings + N blocks + lm_head, tied weights,
                              1/sqrt(2N) residual init                      [DONE]
  Day 6  generate()         - temperature / top-k sampling (extends below)

Phase 3 (Days 13-18) upgrades to a Llama-style stack: RoPE, RMSNorm, SwiGLU,
KV cache, grouped-query attention.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated and projected."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP with the standard 4x hidden expansion."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm(nn.Module):
    """LayerNorm from scratch, numerically identical to nn.LayerNorm.
    Biased variance (unbiased=False), normalize over the last dim only."""

    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))
        self.beta = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta


class Block(nn.Module):
    """Pre-norm transformer block:
        x = x + MHA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """A decoder-only GPT: token + positional embeddings, a stack of pre-norm
    blocks, a final LayerNorm, and a tied LM head."""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size,
                 dropout=0.0):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # init everything, THEN tie + scale (so tying isn't clobbered by apply)
        self.apply(self._init_weights)

        # weight tying: the input embedding and output projection share weights.
        self.lm_head.weight = self.token_embedding.weight

        # GPT-2 residual init: scale the projections that write back into the
        # residual stream by 1/sqrt(2N) so variance doesn't compound with depth.
        scale = 0.02 * (2 * n_layer) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("sa.proj.weight") or name.endswith("ffwd.net.2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                   # (B,T,C)
        pos = self.position_embedding(torch.arange(T, device=idx.device)) # (T,C)
        x = self.drop(tok + pos)                                          # broadcast
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                          # (B,T,vocab)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]              # last step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
