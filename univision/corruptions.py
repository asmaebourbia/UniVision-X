"""Geometry-preserving corruptions. Images [T,3,H,W], float in [0,1]."""
import hashlib
import random
import torch
import torch.nn.functional as F


def deterministic_seed(*items):
    key = "::".join(map(str, items)).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def gaussian_blur(img, sigma=1.5):
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    kernel = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    k2 = torch.outer(kernel, kernel)
    k2 = k2.view(1, 1, *k2.shape).repeat(3, 1, 1, 1)
    return F.conv2d(F.pad(img, (radius,) * 4, mode="reflect"), k2, groups=3)


def corrupt(x, mode="clean", severity=2, seed=None):
    """x is a *clip*. Deterministic only when seed is provided."""
    if mode == "clean":
        return x.clone()
    sev = float(severity)
    if mode == "blur":
        return gaussian_blur(x, sigma=0.6 + 0.6 * sev).clamp(0, 1)
    if mode == "fog":
        return (x * (1 - 0.09 * sev) + (0.09 * sev)).clamp(0, 1)
    if mode == "dark":
        return (x * max(0.12, 1 - 0.19 * sev)).clamp(0, 1)
    if mode == "noise":
        gen = torch.Generator(device=x.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
        return (x + noise * (0.025 * sev)).clamp(0, 1)
    raise ValueError(f"Unknown corruption: {mode}")


def random_corruption(x, probability=0.7):
    if random.random() >= probability:
        return x.clone()
    return corrupt(x, random.choice(["noise", "blur", "fog", "dark"]), random.randint(1, 3))
