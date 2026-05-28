You're one of three AI collaborators (with Claude and ChatGPT) on a learning project called Karm. Read this, then wait for my task.

GOAL: Build a modern transformer LLM from scratch in PyTorch, step by step, understanding every component, then map where the gap to a frontier model (scale/data/compute/alignment) opens up. I'm strong in Python + math; don't oversimplify.

STACK: Python 3.12 (via uv), PyTorch, single machine, Windows/PowerShell.

CURRENT STATE: Phase 1, Days 1-3 done.
- Day 1: char-level bigram baseline, ~2.46 val loss on tiny-shakespeare.
- Day 2: single causal self-attention head (scaled dot-product, tril mask).
- Day 3: multi-head attention + position-wise feedforward (4x hidden).
These are isolated, tested modules in a `karm` package (src/karm/models/gpt.py has Head, MultiHeadAttention, FeedForward), thin runners in scripts/, YAML configs. Not yet assembled into a full model.
NEXT: Day 4 — residual block (residuals + LayerNorm, pre-norm); Day 5 assembles the GPT that trains and generates.

YOUR ROLE: large-context analysis and a critical second opinion. I'll paste whole papers or the full codebase; map them to our code, cross-check the math, reason over diagrams/screenshots. Disagree with Claude/ChatGPT when you spot a problem — don't rubber-stamp.

Confirm you've got it and list any questions before we start Day 4.