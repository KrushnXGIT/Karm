"""The real GPT, assembled across Days 2-6. Placeholder until then.

Build order (each day adds one class, tested in isolation first):
  Day 2  Head               - single scaled dot-product self-attention head
  Day 3  MultiHeadAttention - parallel heads + projection
         FeedForward        - position-wise MLP
  Day 4  Block              - attn + ff with residuals & LayerNorm (pre-norm)
  Day 5  GPT                - token + positional embeddings, N blocks, lm_head
  Day 6  generate()         - sampling with temperature / top-k

Phase 3 (Days 13-18) then upgrades this to a modern Llama-style stack:
RoPE, RMSNorm, SwiGLU, KV cache, grouped-query attention.
"""
