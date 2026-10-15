"""MeViS v2 loader. Train/valid_u have COCO RLE masks; valid does not.

Each example is one referring expression; masks for multiple anno_id objects
are UNIONED, including valid empty/no-target expressions.
"""
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .corruptions import corrupt, random_corruption, deterministic_seed


def decode_rle(annotation):
    if annotation is None:
        return None
    # COCO supports uncompressed counts lists; this pure-Python path makes
    # synthetic smoke tests possible without compiled pycocotools.
    if isinstance(annotation, dict) and isinstance(annotation.get("counts"), list):
        H, W = annotation["size"]
        vals = []
        bit = 0
        for run in annotation["counts"]:
            vals.extend([bit] * int(run))
            bit = 1 - bit
        if len(vals) != H * W:
            raise ValueError("Invalid uncompressed RLE length")
        return np.array(vals, dtype=np.uint8).reshape((H, W), order="F")
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise ImportError("MeViS COCO RLE requires: pip install pycocotools") from exc
    out = coco_mask.decode(annotation)
    if out.ndim == 3:
        out = (out > 0).any(axis=-1)
    return (out > 0).astype(np.uint8)


def normal_text(text):
    return " ".join(text.lower().strip().split())


class MeViS(Dataset):
    def __init__(self, root, split="train", image_size=256, clip_length=4,
                 full_video=False, train=False, corrupt_mode="clean", corruption_probability=0.7,
                 text_cache=None, visual_only=False, teacher_mask_root=None, limit=0):
        self.split_root = Path(root) / split
        with open(self.split_root / "meta_expressions.json", encoding="utf-8") as f:
            videos = json.load(f)["videos"]
        gt_path = self.split_root / "mask_dict.json"
        if not gt_path.exists():
            raise ValueError(f"Split {split} has no mask_dict.json; use train or valid_u for evaluation")
        with open(gt_path, encoding="utf-8") as f:
            self.mask_dict = json.load(f)
        self.records = []
        for video, data in sorted(videos.items()):
            frames = sorted(data["frames"])
            for exp_id, expr in sorted(data["expressions"].items()):
                self.records.append({
                    "video": video, "exp_id": str(exp_id), "frames": frames,
                    "text": normal_text(expr["exp"]),
                    "ann_ids": [str(a) for a in (expr.get("anno_id") or [])],
                })
        if limit > 0:
            self.records = self.records[:limit]
        self.image_size = int(image_size)
        self.clip_length = int(clip_length)
        self.full_video = bool(full_video)
        self.train_mode = bool(train)
        self.corrupt_mode = corrupt_mode
        self.corruption_probability = float(corruption_probability)
        self.visual_only = visual_only
        self.teacher_mask_root = Path(teacher_mask_root) if teacher_mask_root else None
        if visual_only:
            self.text_cache = {}
        else:
            if not text_cache:
                raise ValueError("Provide CLIP embeddings via --text-cache, or explicitly set --visual-only")
            self.text_cache = torch.load(text_cache, map_location="cpu", weights_only=True)
        if not self.records:
            raise ValueError(f"No expressions found in {self.split_root}")

    def __len__(self):
        return len(self.records)

    def _indices(self, n):
        if self.full_video:
            return list(range(n))
        if not self.train_mode:
            return [min(i, n-1) for i in range(self.clip_length)]
        step = random.choice([1, 1, 2])
        start_max = max(0, n - (self.clip_length - 1) * step - 1)
        start = random.randint(0, start_max)
        return [min(start + i * step, n-1) for i in range(self.clip_length)]

    def __getitem__(self, idx):
        r = self.records[idx]
        indices = self._indices(len(r["frames"]))
        imgs, masks, teacher, teacher_available = [], [], [], []
        orig_h, orig_w = None, None
        for index in indices:
            name = r["frames"][index]
            impath = self.split_root / "JPEGImages" / r["video"] / f"{name}.jpg"
            with Image.open(impath) as im:
                rgb = im.convert("RGB")
                orig_w, orig_h = rgb.size
                img = rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                arr = np.asarray(img).copy()
                imgs.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.)
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for ann_id in r["ann_ids"]:
                rles = self.mask_dict.get(ann_id)
                if rles is None or index >= len(rles):
                    continue
                decoded = decode_rle(rles[index])
                if decoded is not None:
                    if decoded.shape != mask.shape:
                        raise ValueError(f"RLE mask size mismatch: {r['video']} {name}")
                    mask = np.logical_or(mask, decoded).astype(np.uint8)
            m = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
            masks.append(torch.from_numpy(np.asarray(m).copy())[None].float())
            if self.teacher_mask_root:
                pp = self.teacher_mask_root / r["video"] / r["exp_id"] / f"{name}.png"
                if pp.exists():
                    with Image.open(pp) as pm:
                        arr = np.asarray(pm.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)).copy()
                    teacher.append(torch.from_numpy((arr > 127).astype(np.float32))[None])
                    teacher_available.append(True)
                else:
                    teacher.append(torch.zeros_like(masks[-1]))
                    teacher_available.append(False)
        clean = torch.stack(imgs)
        if self.train_mode:
            altered = random_corruption(clean, self.corruption_probability)
        else:
            altered = corrupt(clean, self.corrupt_mode, severity=2,
                              seed=deterministic_seed(r["video"], r["exp_id"], self.corrupt_mode))
        if self.visual_only:
            text_embedding = torch.zeros(512)
        else:
            if r["text"] not in self.text_cache:
                raise KeyError(f"Text not in CLIP cache: {r['text']!r}. Recompute cache for this split.")
            text_embedding = self.text_cache[r["text"]].float()
        return {
            "video": altered, "clean_video": clean,
            "mask": torch.stack(masks), "text_emb": text_embedding,
            "text": r["text"], "name": r["video"], "exp_id": r["exp_id"],
            "frame_names": [r["frames"][i] for i in indices],
            "orig_hw": (orig_h, orig_w),
            "teacher_mask": torch.stack(teacher) if teacher else torch.zeros_like(torch.stack(masks)),
            "teacher_valid": torch.tensor(teacher_available, dtype=torch.bool) if teacher_available else torch.zeros(len(indices), dtype=torch.bool),
        }
