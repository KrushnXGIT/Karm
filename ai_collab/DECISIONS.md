# Decisions log (lightweight ADRs)

Format: date — decision — why — revisit-when.

- 2026-05-28 — Character-level tokenizer to start. Simplest possible; isolates
  the modeling problem from the tokenization problem. Revisit: Day 7 (BPE).
- 2026-05-28 — `src/` package layout with thin `scripts/` runners. Forces code
  reuse as the project grows past ~30 files. Revisit: unlikely.
- 2026-05-28 — Pre-norm transformer blocks (LayerNorm before attn/ff) planned
  for Day 4 — more stable training than post-norm. Revisit: if exploring norms.
