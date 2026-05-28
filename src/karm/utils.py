"""Shared utilities: device, seeding, parameter counts, checkpoints."""
import torch


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, path, **extra):
    torch.save({"model_state": model.state_dict(), **extra}, path)


def load_checkpoint(model, path, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    return ckpt
