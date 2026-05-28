# Progress journal

## Day 1 — 2026-05-28
- Built char-level bigram LM; training loop + estimate_loss + sampling.
- Refactored monolith into karm package; added pytest sanity tests.
- Result: val loss ~2.46, output is character-statistically plausible gibberish.
- Takeaway: one-token context is the hard ceiling -> motivates attention.

## Day 2 — single self-attention head
- Built up attention in 4 steps: uniform average -> matmul -> softmax -> QKV.
- Confirmed v1==v2==v3 numerically; attention = data-dependent averaging.
- Added Head (scaled dot-product, causal tril mask, dropout) to models/gpt.py.
- Takeaway: the 1/sqrt(d_k) scale and causal mask are load-bearing, not cosmetic.

## Day 3 — multi-head attention + feedforward
- Added MultiHeadAttention (n_head parallel heads of size n_embd//n_head, then
  concat + projection) and FeedForward (4x hidden MLP).
- Verified shape preserved (B,T,n_embd) in==out -> modules are stackable.
- Takeaway: attention = communication, feedforward = computation; the FFN holds
  most of the parameters.

## Infra note — 2026-05-28
- Switched Python 3.14 -> 3.12 and adopted uv (.python-version pinned).
- Next: Day 4 — residual block (residuals + LayerNorm).
