"""Tiny moving-shape MeViS-shaped data for integration tests (NOT a benchmark)."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from .utils import ensure_dir, write_json
from .data import normal_text


def rle_encode_uncompressed(mask):
    flat = mask.astype(np.uint8).ravel(order="F")
    runs = []
    current = 0
    count = 0
    for value in flat:
        if value == current:
            count += 1
        else:
            runs.append(count)
            current = int(value)
            count = 1
    runs.append(count)
    return {"size": list(mask.shape), "counts": runs}


def make_dataset(root, image_size=96, n_train=8, n_val=3, nframes=6):
    root = Path(root)
    texts = ["the red square moving right", "the blue circle moving left"]
    for split, nvid in [("train", n_train), ("valid_u", n_val)]:
        splitdir = root / split
        videos, masks = {}, {}
        for vi in range(nvid):
            vid = f"vid_{vi:03d}"
            folder = splitdir / "JPEGImages" / vid
            ensure_dir(folder)
            frame_names = []
            for obj in range(2):
                masks[f"{vid}_obj_{obj}"] = []
            for t in range(nframes):
                name = f"{t:05d}"
                frame_names.append(name)
                pic = Image.new("RGB", (image_size, image_size), (24, 28, 33))
                pdraw = ImageDraw.Draw(pic)
                for obj in range(2):
                    mr = Image.new("L", (image_size, image_size), 0)
                    md = ImageDraw.Draw(mr)
                    span = max(1, image_size - 32)
                    if obj == 0:
                        x = 6 + (t * 3 + vi * 5) % span
                        y = 15 + (vi * 2) % 8
                        shape = [x, y, x + 19, y + 19]
                        pdraw.rectangle(shape, fill=(225, 55, 55))
                        md.rectangle(shape, fill=1)
                    else:
                        x = image_size - 30 - (t * 3 + vi * 3) % span
                        y = image_size - 31
                        shape = [x, y, x + 18, y + 18]
                        pdraw.ellipse(shape, fill=(49, 110, 220))
                        md.ellipse(shape, fill=1)
                    masks[f"{vid}_obj_{obj}"].append(rle_encode_uncompressed(np.asarray(mr)))
                pic.save(folder / f"{name}.jpg", quality=95)
            videos[vid] = {"frames": frame_names, "expressions": {
                str(i): {"exp": text, "anno_id": [f"{vid}_obj_{i}"], "obj_id": [i+1]}
                for i, text in enumerate(texts)
            }}
        write_json({"videos": videos}, splitdir / "meta_expressions.json")
        write_json(masks, splitdir / "mask_dict.json")
    # Test embeddings are arbitrary and do NOT convey language semantics.
    emb = {normal_text(t): torch.randn(512, generator=torch.Generator().manual_seed(100+i)) for i, t in enumerate(texts)}
    ensure_dir(root)
    torch.save(emb, root / "SYNTHETIC_TEXT_ONLY.pt")
    print(f"Synthetic data: {root} ; synthetic nonsemantic vectors: {root / 'SYNTHETIC_TEXT_ONLY.pt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--train-videos", type=int, default=8)
    ap.add_argument("--val-videos", type=int, default=3)
    a = ap.parse_args()
    make_dataset(a.out, image_size=a.image_size, n_train=a.train_videos, n_val=a.val_videos)
