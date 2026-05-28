"""Generation helpers."""
import torch


def generate_text(model, tokenizer, device, max_new_tokens=400, prompt=""):
    """Autoregressively sample text from a trained model."""
    if prompt:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0].tolist())
