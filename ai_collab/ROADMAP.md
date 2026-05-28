# Roadmap — 5 phases

## Phase 1 — Core transformer (Days 1-6)
1. Data, char tokenizer, bigram baseline        [DONE]
2. Self-attention from scratch (single head)
3. Multi-head attention + feedforward
4. Transformer block: residuals + LayerNorm
5. Assemble + train a small GPT, real samples
6. Sampling: temperature, top-k

## Phase 2 — Real system (Days 7-12)
BPE tokenizer; data pipeline & sharding; AdamW + warmup/cosine LR, grad
clipping, mixed precision, gradient accumulation; eval (loss, perplexity).

## Phase 3 — Modern architecture (Days 13-18)
RoPE; RMSNorm; SwiGLU; KV cache; grouped-query attention; init/weight tying.

## Phase 4 — Scaling & data (Days 19-24)
Chinchilla scaling laws; data curation & dedup; DDP/FSDP/parallelism concepts;
where a single GPU stops being enough.

## Phase 5 — Post-training (Days 25-30)
SFT / instruction tuning; preference data; DPO and RLHF; why a small aligned
model can feel more useful than a huge raw one.

Rule at every step: working code you run + one honest line on the frontier gap.
