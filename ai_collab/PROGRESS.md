# Progress journal

## Day 1 — 2026-05-28
- Built char-level bigram LM; training loop + estimate_loss + sampling.
- Refactored monolith into karm package; added pytest sanity tests.
- Result: val loss ~2.45, output is character-statistically plausible gibberish.
- Takeaway: one-token context is the hard ceiling -> motivates attention.
- Next: Day 2, single self-attention head.
