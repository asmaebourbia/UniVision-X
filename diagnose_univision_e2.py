#!/usr/bin/env python3
"""Post-hoc diagnosis for UniVision-X MeViS valid_u evaluation.

Does not modify weights or run training. Produces a CSV-metric report and
optional qualitative comparisons (RGB / ground truth / predicted segmentation).

Example:
python diagnose_univision_e2.py \
 --eval outputs/E2_eval_clean \
 --data data/MeViS_release/valid_u \
 --out outputs/E2_diagnostics
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def binary_rle_mask(rle):
    if rle is None:
        return None
    if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
        h, w = rle["size"]
        a = np.zeros(h * w, dtype=np.uint8)
        cursor = 0
        value = 0
        for run in rle["counts"]:
            stop = cursor + int(run)
            if value:
                a[cursor:stop] = 1
            cursor = stop
            value = 1 - value
        if cursor != h * w:
            raise ValueError("Uncompressed RLE length differs from image dimensions")
        return a.reshape((h, w), order="F").astype(bool)
    from pycocotools import mask as coco_mask
    data = coco_mask.decode(rle)
    if data.ndim == 3:
        data = data.any(axis=-1)
    return data.astype(bool)


def mask_for_frame(ann_ids, mask_dict, index, h, w):
    target = np.zeros((h, w), dtype=bool)
    for ann_id in ann_ids:
        sequence = mask_dict.get(str(ann_id))
        if sequence and index < len(sequence):
            m = binary_rle_mask(sequence[index])
            if m is not None:
                if m.shape != target.shape:
                    raise ValueError("Ground truth/image dimensions mismatch")
                target |= m
    return target


def pct(q, values):
    if not values:
        return None
    return float(np.percentile(values, q))


def overlay(rgb, binary, color):
    rgb = rgb.convert("RGB")
    arr = np.asarray(rgb).copy().astype(np.float32)
    binary = np.asarray(binary, dtype=bool)
    tint = np.asarray(color, dtype=np.float32)
    arr[binary] = 0.48 * arr[binary] + 0.52 * tint
    return Image.fromarray(np.uint8(np.clip(arr, 0, 255)))


def one_case(row, split_root, videos, masks, eval_root, output):
    vid = row["video"]
    exp_id = row["exp_id"]
    record = videos[vid]
    expr = record["expressions"][exp_id]
    frame_names = sorted(record["frames"])
    candidates = sorted(set(np.linspace(0, len(frame_names) - 1, min(11, len(frame_names)), dtype=int).tolist()))
    index = candidates[len(candidates)//2]
    best = None
    for j in candidates:
        name = frame_names[j]
        source = split_root / "JPEGImages" / vid / f"{name}.jpg"
        pred_path = eval_root / "Annotations" / vid / exp_id / f"{name}.png"
        if not source.exists() or not pred_path.exists():
            continue
        with Image.open(source) as im:
            w, h = im.size
        gt = mask_for_frame(expr.get("anno_id") or [], masks, j, h, w)
        # Prefer a frame with a visible target, otherwise pick a valid frame.
        if best is None or (not best[2].any() and gt.any()):
            best = (j, name, gt)
        if gt.any():
            break
    if best is None:
        raise FileNotFoundError(f"Missing source or predicted frames for {vid}/{exp_id}")
    index, name, gt = best
    with Image.open(split_root / "JPEGImages" / vid / f"{name}.jpg") as im:
        rgb = im.convert("RGB")
    with Image.open(eval_root / "Annotations" / vid / exp_id / f"{name}.png") as pm:
        pred = np.asarray(pm.convert("L").resize(rgb.size, Image.Resampling.NEAREST)) > 127
    panels = [("Original RGB", rgb), ("Annotation GT", overlay(rgb, gt, (45, 210, 80))),
              ("Prediction", overlay(rgb, pred, (240, 70, 60)))]
    W, H = 384, 272
    canvas = Image.new("RGB", (3*W, H+92), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    for k, (label, pic) in enumerate(panels):
        shown = ImageOps.contain(pic, (W-16, H-10))
        x = k*W + (W-shown.width)//2
        y = 28 + (H-shown.height)//2
        canvas.paste(shown, (x,y))
        draw.text((k*W+10, 8), label, fill=(15,15,15))
    draw.text((10, H+34), f"video={vid} exp_id={exp_id} frame={name} J={float(row['J']):.3f}", fill=(0,0,0))
    desc = row.get("text", "")
    draw.text((10, H+56), "Expression: " + desc[:145], fill=(0,0,0))
    outfile = output / f"case_{vid}_{exp_id}.png"
    canvas.save(outfile)
    return str(outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("outputs/E2_eval_clean"))
    ap.add_argument("--data", type=Path, default=Path("data/MeViS_release/valid_u"))
    ap.add_argument("--out", type=Path, default=Path("outputs/E2_diagnostics"))
    ap.add_argument("--no-visuals", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.eval / "per_expression.csv", newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError("Empty per_expression.csv")
    js = [float(r["J"]) for r in records]
    dice = [float(r["Dice"]) for r in records]
    report = {
        "n_expressions": len(records),
        "J_mean": statistics.mean(js), "J_median": statistics.median(js),
        "J_p10": pct(10, js), "J_p90": pct(90, js),
        "Dice_mean": statistics.mean(dice),
        "n_zero_J": sum(x == 0 for x in js),
        "n_J_below_0_1": sum(x < 0.1 for x in js),
        "n_J_above_or_equal_0_5": sum(x >= 0.5 for x in js),
        "n_J_above_or_equal_0_7": sum(x >= 0.7 for x in js),
        "frames_mean": statistics.mean(int(r["frames"]) for r in records),
        "fps_mean_per_expression_excludes_data_loading": statistics.mean(float(r["fps"]) for r in records),
    }
    with open(args.data / "meta_expressions.json", encoding="utf-8") as f:
        videos = json.load(f)["videos"]
    # This distinguishes expressions without annotation IDs from others;
    # it does not claim that all frames of an annotated expression have a target.
    no_ann = []
    annotated = []
    for row in records:
        expr = videos[row["video"]]["expressions"][row["exp_id"]]
        (annotated if (expr.get("anno_id") or []) else no_ann).append(float(row["J"]))
    report["no_annotation_ids_count"] = len(no_ann)
    report["no_annotation_ids_mean_J"] = statistics.mean(no_ann) if no_ann else None
    report["has_annotation_ids_mean_J"] = statistics.mean(annotated) if annotated else None
    if not args.no_visuals:
        with open(args.data / "mask_dict.json", encoding="utf-8") as f:
            masks = json.load(f)
        ranked = sorted(records, key=lambda r: float(r["J"]))
        desired = [0, len(ranked)//4, len(ranked)//2, 3*len(ranked)//4, len(ranked)-1]
        chosen = [ranked[x] for x in dict.fromkeys(desired)]
        paths = []
        for r in chosen:
            try:
                paths.append(one_case(r, args.data, videos, masks, args.eval, args.out))
            except Exception as exc:
                print("Skipping qualitative case", r["video"], r["exp_id"], str(exc))
        report["qualitative_cases"] = paths
    (args.out / "diagnosis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
