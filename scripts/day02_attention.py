"""Day 2 — Self-attention from scratch, built up in four steps.

The point of this script is to SEE that "averaging past tokens" and
"self-attention" are the same operation, where attention just makes the
averaging weights data-dependent. Steps 1-3 produce identical numbers; step 4
is the real thing.

Run from the repo root:  python scripts/day02_attention.py
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 32  # batch, time (context length), channels (embedding dim)
x = torch.randn(B, T, C)

# ---------------------------------------------------------------------------
# v1: bag-of-words. Each token = mean of itself + all PREVIOUS tokens.
#     Explicit loops so the operation is unambiguous.
# ---------------------------------------------------------------------------
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t + 1]        # (t+1, C) -- only the past, inclusive
        xbow[b, t] = xprev.mean(dim=0)

# ---------------------------------------------------------------------------
# v2: the same average expressed as a single matrix multiply.
#     A lower-triangular matrix of ones, row-normalized, IS "average the past".
#     tril -> can't see the future. Normalizing each row -> it's a mean.
# ---------------------------------------------------------------------------
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)   # each row sums to 1
xbow2 = wei @ x                             # (T,T) @ (B,T,C) -> (B,T,C)
assert torch.allclose(xbow, xbow2, atol=1e-6)

# ---------------------------------------------------------------------------
# v3: the same average via softmax. Start at zeros, forbid the future with
#     -inf, softmax. softmax(0 over allowed positions) = uniform = the mean.
#     Why bother? Because now the weights COME FROM softmax of some scores --
#     and those scores no longer have to be zero. That's the door to v4.
# ---------------------------------------------------------------------------
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float("-inf"))  # future -> -inf
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
assert torch.allclose(xbow, xbow3, atol=1e-6)

print("v1 == v2 == v3  ->  a uniform average is just a masked, softmaxed matmul")

# ---------------------------------------------------------------------------
# v4: SELF-ATTENTION. The weights stop being uniform and become data-dependent.
#     Each token emits a query (what it's looking for) and a key (what it holds).
#     affinity = query . key. Scale, mask the future, softmax -> weights.
#     Then aggregate each token's VALUE with those weights.
# ---------------------------------------------------------------------------
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)     # (B, T, head_size)
q = query(x)   # (B, T, head_size)

# scores: how much each query matches each key. Scale by 1/sqrt(head_size)
# so the dot products don't blow up and saturate softmax at init.
wei = q @ k.transpose(-2, -1) * head_size ** -0.5   # (B, T, T)
wei = wei.masked_fill(tril == 0, float("-inf"))      # causal mask
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v   # (B, T, head_size)

print(f"v4 self-attention output shape: {tuple(out.shape)}  (B, T, head_size)")
print("\nattention weights for batch 0 (rows=query token, cols=attended token):")
torch.set_printoptions(precision=2, sci_mode=False)
print(wei[0])
# Note how row 0 attends only to token 0, row 1 splits over tokens 0-1, etc.
# Lower-triangular by construction, and NOT uniform -- that's it learning
# (well, would learn -- these projections are random here) what to look at.
