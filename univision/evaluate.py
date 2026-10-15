"""Full-sequence local evaluation on MeViS valid_u. Writes MeViS-compatible PNGs.

For official leaderboard J&F use the MeViS authors' evaluation script on the saved PNGs.
Our local boundary F is an explicitly approximate fixed-pixel-tolerance proxy.
"""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from .data import MeViS
from .model import UniVisionX
from .metrics import mask_iou, mask_dice, boundary_f, uncertainty_mean
from .utils import ensure_dir, available_device, write_json


def build_checkpoint_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    a = ckpt["config"]
    model = UniVisionX(encoder=a["encoder"], pretrained=False,
                       width=a["width"], use_text=not a["visual_only"],
                       use_temporal=not a["no_temporal"],
                       freeze_encoder=a["freeze_encoder"], lora_rank=a["lora_rank"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), a


@torch.inference_mode()
def inference_clip(model, video, text, chunk=4, use_amp=False):
    """Processes an entire expression sequentially; GRU state travels across chunks."""
    device = next(model.parameters()).device
    h = None
    outputs = []
    for i in range(0, len(video), chunk):
        vi = video[i:i+chunk][None].to(device)
        te = text[None].to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits, h = model(vi, te, hidden=h)
        if h is not None:
            h = h.detach()
        outputs.append(logits.squeeze(0).float().cpu())
    return torch.cat(outputs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--split", default="valid_u")
    ap.add_argument("--corruption", choices=["clean", "noise", "blur", "fog", "dark"], default="clean")
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-png", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be >=1")
    device = available_device()
    model, config = build_checkpoint_model(args.checkpoint, device)
    dataset = MeViS(args.data, args.split, config["image_size"], full_video=True,
                    train=False, corrupt_mode=args.corruption,
                    text_cache=args.text_cache, visual_only=config["visual_only"], limit=args.limit)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    results = []
    for i in tqdm(range(len(dataset)), desc="full video eval"):
        sample = dataset[i]
        t0 = time.perf_counter()
        logits = inference_clip(model, sample["video"], sample["text_emb"],
                                chunk=args.chunk_size, use_amp=args.amp)
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - t0
        preds = (logits.sigmoid() > .5).squeeze(1).numpy()
        target = sample["mask"].squeeze(1).numpy() > .5
        J = float(np.mean([mask_iou(p, g) for p, g in zip(preds, target)]))
        Dice = float(np.mean([mask_dice(p, g) for p, g in zip(preds, target)]))
        Fscore = float(np.mean([boundary_f(p, g) for p, g in zip(preds, target)]))
        rec = {"video": sample["name"], "exp_id": sample["exp_id"],
               "text": sample["text"], "frames": len(preds), "J": J, "Dice": Dice,
               "boundary_F_proxy": Fscore, "J_and_proxy_F": (J+Fscore)/2,
               "entropy": uncertainty_mean(logits), "seconds": round(duration, 4),
               "fps": len(preds)/max(1e-6, duration)}
        results.append(rec)
        if args.save_png:
            # MeViS submission directory layout; predictions are upsampled to source resolution.
            folder = out_dir / "Annotations" / sample["name"] / sample["exp_id"]
            ensure_dir(folder)
            oh, ow = sample["orig_hw"]
            for mask, name in zip(preds, sample["frame_names"]):
                img = Image.fromarray((mask*255).astype(np.uint8)).resize((ow, oh), Image.Resampling.NEAREST)
                img.save(folder / f"{name}.png")
        if (i + 1) % 20 == 0:
            print("Processed", i+1, "expressions")
    averages = {k: float(np.mean([r[k] for r in results])) for k in
                ["J", "Dice", "boundary_F_proxy", "J_and_proxy_F", "entropy", "fps"]}
    summary = {"split": args.split, "corruption": args.corruption, "n_expressions": len(results),
               "local_metrics": averages,
               "warning": "These are resized local metrics; boundary F uses a fixed-pixel proxy, NOT official MeViS J&F. Use the official evaluation code on the exported PNGs."}
    write_json(summary, out_dir / "summary.json")
    with open(out_dir / "per_expression.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
