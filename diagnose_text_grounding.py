"""Assess whether UniVision-X predictions change with the referring expression.

Use on a *held-out* MeViS split (valid_u). For each sampled expression, keep the
video unchanged, replace only its CLIP text embedding with a different expression
(preferably from the same video), and compare masks and IoU. Never trains model.
Run at repository root: python diagnose_text_grounding.py --help
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from univision.data import MeViS
from univision.evaluate import build_checkpoint_model, inference_clip
from univision.metrics import mask_iou
from univision.utils import available_device


def metrics(pred, target):
    J = np.array([mask_iou(p, g) for p, g in zip(pred, target)], dtype=np.float64)
    present = target.reshape(target.shape[0], -1).any(axis=1)
    fg = pred.reshape(pred.shape[0], -1).any(axis=1)
    return {
        "J": float(J.mean()),
        "J_present": float(J[present].mean()) if present.any() else None,
        "target_absent_frames": int((~present).sum()),
        "false_positive_on_absent": int((fg & ~present).sum()),
        "predicted_coverage": float(pred.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--out", default="outputs/E2_text_diagnostic")
    parser.add_argument("--split", default="valid_u")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    a = parser.parse_args()
    if a.limit <= 0 or a.chunk_size < 1:
        parser.error("--limit and --chunk-size must be positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = available_device()
    model, config = build_checkpoint_model(a.checkpoint, device)
    if config["visual_only"]:
        raise ValueError("The checkpoint is visual-only; no text-grounding test is possible")
    dataset = MeViS(a.data, split=a.split, image_size=config["image_size"],
                    full_video=True, train=False, text_cache=a.text_cache)
    video_to_indices = defaultdict(list)
    for i, rec in enumerate(dataset.records):
        video_to_indices[rec["video"]].append(i)
    rng = random.Random(a.seed)
    selection = rng.sample(range(len(dataset)), min(a.limit, len(dataset)))
    rows = []
    for done, idx in enumerate(selection, 1):
        rec = dataset.records[idx]
        same_video = [j for j in video_to_indices[rec["video"]]
                      if dataset.records[j]["text"] != rec["text"]]
        if same_video:
            other_idx = rng.choice(same_video)
            negative_type = "same_video"
        else:
            candidates = [j for j in range(len(dataset)) if dataset.records[j]["text"] != rec["text"]]
            if not candidates:
                raise ValueError("Need at least two distinct expressions")
            other_idx = rng.choice(candidates)
            negative_type = "different_video"
        example = dataset[idx]
        wrong_rec = dataset.records[other_idx]
        wrong_embedding = dataset.text_cache[wrong_rec["text"]].float()
        with torch.inference_mode():
            logits_correct = inference_clip(model, example["video"], example["text_emb"],
                                            chunk=a.chunk_size, use_amp=a.amp)
            logits_wrong = inference_clip(model, example["video"], wrong_embedding,
                                          chunk=a.chunk_size, use_amp=a.amp)
        prob_correct = logits_correct.sigmoid().squeeze(1).numpy()
        prob_wrong = logits_wrong.sigmoid().squeeze(1).numpy()
        p_correct = prob_correct > 0.5
        p_wrong = prob_wrong > 0.5
        target = example["mask"].squeeze(1).numpy() > 0.5
        m_correct = metrics(p_correct, target)
        m_wrong = metrics(p_wrong, target)
        rows.append({
            "video": rec["video"], "exp_id": rec["exp_id"],
            "true_text": rec["text"], "wrong_text": wrong_rec["text"],
            "negative_type": negative_type, "frames": int(len(p_correct)),
            "J_correct": m_correct["J"], "J_wrong": m_wrong["J"],
            "J_correct_present": m_correct["J_present"],
            "J_wrong_present": m_wrong["J_present"],
            "J_gain_correct_vs_wrong": m_correct["J"] - m_wrong["J"],
            "changed_mask_fraction": float((p_correct != p_wrong).mean()),
            "mean_abs_probability_change": float(np.abs(prob_correct-prob_wrong).mean()),
            "gt_absent_frames": m_correct["target_absent_frames"],
            "false_positive_frames_correct": m_correct["false_positive_on_absent"],
            "false_positive_frames_wrong": m_wrong["false_positive_on_absent"],
            "pred_coverage_correct": m_correct["predicted_coverage"],
            "pred_coverage_wrong": m_wrong["predicted_coverage"],
        })
        if done % 10 == 0:
            print(f"Processed {done}/{len(selection)} expressions", flush=True)
    with open(out / "per_expression.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def avg(k):
        vals = [r[k] for r in rows if r[k] is not None]
        return float(np.mean(vals)) if vals else None
    summary = {
        "n_expressions": len(rows),
        "n_same_video_negatives": sum(r["negative_type"] == "same_video" for r in rows),
        "J_correct_mean": avg("J_correct"),
        "J_wrong_mean": avg("J_wrong"),
        "J_gain_correct_vs_wrong_mean": avg("J_gain_correct_vs_wrong"),
        "changed_mask_fraction_mean": avg("changed_mask_fraction"),
        "probability_change_mean": avg("mean_abs_probability_change"),
        "J_correct_present_mean": avg("J_correct_present"),
        "J_wrong_present_mean": avg("J_wrong_present"),
        "absent_target_frames_total": sum(r["gt_absent_frames"] for r in rows),
        "false_positive_frames_correct_total": sum(r["false_positive_frames_correct"] for r in rows),
        "explanation": "Paired text-swap diagnostic, not an official MeViS score. A large margin suggests text is useful; small changes are evidence of weak language sensitivity but are not conclusive alone. Different descriptions can occasionally refer to the same target.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
