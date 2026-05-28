"""Reusable training loop and loss estimation — model-agnostic."""
import torch

from .data import get_batch


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size,
                  device, eval_iters=200):
    """Average loss over several batches per split (less noisy than one step)."""
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_model(model, train_data, val_data, *, block_size, batch_size,
                max_iters, eval_interval, eval_iters, learning_rate, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, block_size,
                                   batch_size, device, eval_iters)
            print(f"step {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f}")
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model
