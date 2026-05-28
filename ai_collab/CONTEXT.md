# CONTEXT — read this first
# STATE: Phase 1, Day 3 complete  (append your latest commit hash here)

> Single source of truth for the project. Before asking ANY of the three
> assistants (Claude / Gemini / ChatGPT) for help, paste this file in.
> After anything important changes, update it. This keeps three memoryless
> assistants in sync.

## Goal
Build a modern transformer LLM from scratch in PyTorch, understand every
component, and map precisely where the gap to a frontier model
(scale/data/compute/alignment) opens up. Owner is strong in Python + math;
do not oversimplify.

## Current state
- Phase 1, Days 1-3 done.
  - Day 1: char-level bigram baseline, ~2.46 val loss on tiny-shakespeare.
  - Day 2: single causal self-attention head (scaled dot-product + tril mask).
  - Day 3: multi-head attention + position-wise feedforward (4x hidden).
- These are isolated, individually tested modules in the `karm` package
  (src/karm/models/gpt.py: Head, MultiHeadAttention, FeedForward).
  NOT yet assembled into a full trainable model.
- Next: Day 4 — residual Block (residuals + LayerNorm, pre-norm) to make
  blocks stackable; Day 5 assembles the GPT that trains and generates words.

## Key facts an assistant needs
- Stack: Python 3.12 (via uv), PyTorch, Windows/PowerShell, single machine.
- Corpus: tiny-shakespeare, character-level (BPE arrives Day 7).
- Conventions: reusable logic in src/karm/, thin runners in scripts/dayNN_*.py,
  hyperparameters in configs/*.yaml.
- Decisions log: ai_collab/DECISIONS.md. Progress journal: ai_collab/PROGRESS.md.

## How to update
When something changes: edit "Current state", bump the STATE line + commit hash,
add a line to PROGRESS.md, and log any architecture choice in DECISIONS.md.
Keep this file to ~1 page.
