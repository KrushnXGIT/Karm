# CONTEXT — read this first

> This is the single source of truth for the project. Before asking ANY of the
> three assistants (Claude / Gemini / ChatGPT) for help, paste this file in.
> After anything important changes, update it. This is what keeps three
> memoryless assistants in sync.

## Goal
Build a modern transformer LLM from scratch, understand every component, and
map precisely where the gap to a frontier model (scale/data/compute/alignment)
opens up. Owner is strong in Python + math; do not oversimplify.

## Current state
- Phase 1, Day 1 complete: char-level bigram baseline trains, val loss ~2.45.
- Code refactored into the `karm` package (see README layout).
- Next: Day 2 — self-attention from scratch.

## Key facts an assistant needs
- Stack: Python 3.12, PyTorch, Windows (PowerShell), single machine.
- Corpus: tiny-shakespeare (char-level for now; BPE arrives Day 7).
- Conventions: logic in src/karm/, thin runners in scripts/, configs in YAML.
- Decisions log: ai_collab/DECISIONS.md. Progress journal: ai_collab/PROGRESS.md.

## How to update
When something changes, edit "Current state" + add a line to PROGRESS.md, and
log any architecture choice in DECISIONS.md. Keep this file under ~1 page.
