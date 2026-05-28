"""Fast sanity tests — run with `pytest`. Catch shape/loss regressions early."""
import torch

from karm.models import BigramLanguageModel


def test_forward_shapes_and_loss():
    vocab, B, T = 65, 4, 8
    model = BigramLanguageModel(vocab)
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    logits, loss = model(x, y)
    assert logits.shape == (B, T, vocab)
    assert torch.isfinite(loss)


def test_generate_extends_sequence():
    model = BigramLanguageModel(65)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 11)
