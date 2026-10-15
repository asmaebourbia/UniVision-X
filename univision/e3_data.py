"""MeViS dataset with fixed-length CLIP word-level embeddings and uniform val frames."""
from pathlib import Path
import numpy as np
import torch
from .data import MeViS

class MeViSE3(MeViS):
    def __init__(self, *args, token_cache, **kwargs):
        # E2 dataset expects pooled text features. E3 needs per-token vectors instead.
        kwargs.pop("text_cache", None)
        kwargs.pop("visual_only", None)
        super().__init__(*args, text_cache=None, visual_only=True, **kwargs)
        self.token_cache = torch.load(token_cache, map_location="cpu", weights_only=True)
        first = next(iter(self.token_cache.values()))
        if "tokens" not in first or "attention_mask" not in first:
            raise ValueError("Need E3 word-token cache, not E2 pooled clip_text.pt")

    def _indices(self, n):
        if self.full_video:
            return list(range(n))
        if not self.train_mode:
            return np.linspace(0, n-1, self.clip_length).astype(int).tolist()
        return super()._indices(n)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        entry = self.token_cache.get(sample["text"])
        if entry is None:
            raise KeyError(f"Missing word-level CLIP text: {sample['text']}")
        sample["text_tokens"] = entry["tokens"].float()
        sample["text_attention_mask"] = entry["attention_mask"].bool()
        return sample
