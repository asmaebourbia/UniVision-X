#!/usr/bin/env python3
"""E3 paired text-swap diagnostic on MeViS valid_u (local, not official J&F).

Uses the SAME random sampling/negative-selection algorithm and seed as E2's
`diagnose_text_grounding.py`, permitting paired comparisons on the same videos.
No training or checkpoint modification.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from univision.e3_data import MeViSE3
from univision.e3_model import UniVisionE3
from univision.metrics import mask_iou
from univision.utils import available_device


@torch.inference_mode()
def run_video(model, video, tokens, attention_mask, chunk_size=2, amp=True,
              presence_threshold=0.5):
    device = next(model.parameters()).device
    hidden = None
    preds, segmentation_probs, presence_preds = [], [], []
    for start in range(0, len(video), chunk_size):
        chunk = video[start:start + chunk_size].unsqueeze(0).to(device)
        t = tokens.unsqueeze(0).to(device)
        m = attention_mask.unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
            logits, presence_logits, hidden = model(chunk, t, m, hidden)
        ps = logits.float().sigmoid()[0, :, 0]
        pp = presence_logits.float().sigmoid()[0]
        pred = (ps > 0.5) & (pp > presence_threshold)[:, None, None]
        preds.append(pred.cpu().numpy())
        segmentation_probs.append(ps.cpu().numpy())
        presence_preds.append(pp.cpu().numpy())
        hidden = hidden.detach()
    return (np.concatenate(preds), np.concatenate(segmentation_probs),
            np.concatenate(presence_preds))


def measure(pred, target):
    scores = [mask_iou(p, g) for p, g in zip(pred, target)]
    present = target.reshape(len(target), -1).any(axis=1)
    fg = pred.reshape(len(pred), -1).any(axis=1)
    return {
        'J': float(np.mean(scores)),
        'J_present': float(np.mean(np.asarray(scores)[present])) if present.any() else None,
        'target_absent_frames': int((~present).sum()),
        'false_positive_on_absent': int((fg & ~present).sum()),
        'predicted_coverage': float(pred.mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--token-cache', required=True)
    p.add_argument('--out', default='outputs/E3_text_diagnostic')
    p.add_argument('--split', default='valid_u')
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--chunk-size', type=int, default=2)
    p.add_argument('--presence-threshold', type=float, default=0.5)
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()
    if args.limit < 1 or args.chunk_size < 1:
        p.error('--limit and --chunk-size must be >= 1')

    device = available_device()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = UniVisionE3(encoder=cfg['encoder'], width=cfg['width'], pretrained=False,
                        freeze_encoder=cfg['freeze_encoder']).to(device).eval()
    model.load_state_dict(ckpt['model'])
    ds = MeViSE3(args.data, args.split, image_size=cfg['image_size'],
                 full_video=True, train=False, token_cache=args.token_cache)
    by_video = defaultdict(list)
    for idx, rec in enumerate(ds.records):
        by_video[rec['video']].append(idx)
    rng = random.Random(args.seed)
    selected = rng.sample(range(len(ds)), min(args.limit, len(ds)))
    rows = []
    for number, idx in enumerate(selected, 1):
        rec = ds.records[idx]
        same = [other for other in by_video[rec['video']]
                if ds.records[other]['text'] != rec['text']]
        if same:
            wrong_idx = rng.choice(same)
            negative_type = 'same_video'
        else:
            possible = [other for other in range(len(ds))
                        if ds.records[other]['text'] != rec['text']]
            if not possible:
                raise ValueError('Need at least two different expressions')
            wrong_idx = rng.choice(possible)
            negative_type = 'different_video'
        wrong_record = ds.records[wrong_idx]
        sample = ds[idx]
        wrong_emb = ds.token_cache[wrong_record['text']]
        correct, prob_c, presence_c = run_video(
            model, sample['video'], sample['text_tokens'],
            sample['text_attention_mask'], args.chunk_size, args.amp,
            args.presence_threshold)
        wrong, prob_w, presence_w = run_video(
            model, sample['video'], wrong_emb['tokens'].float(),
            wrong_emb['attention_mask'].bool(), args.chunk_size, args.amp,
            args.presence_threshold)
        target = sample['mask'][:, 0].numpy() > 0.5
        mc, mw = measure(correct, target), measure(wrong, target)
        rows.append({
            'video': rec['video'], 'exp_id': rec['exp_id'],
            'true_text': rec['text'], 'wrong_text': wrong_record['text'],
            'negative_type': negative_type, 'frames': len(correct),
            'J_correct': mc['J'], 'J_wrong': mw['J'],
            'J_correct_present': mc['J_present'], 'J_wrong_present': mw['J_present'],
            'J_gain_correct_vs_wrong': mc['J'] - mw['J'],
            'changed_mask_fraction': float((correct != wrong).mean()),
            'mean_abs_probability_change': float(np.abs(prob_c - prob_w).mean()),
            'gt_absent_frames': mc['target_absent_frames'],
            'false_positive_frames_correct': mc['false_positive_on_absent'],
            'false_positive_frames_wrong': mw['false_positive_on_absent'],
            'pred_coverage_correct': mc['predicted_coverage'],
            'pred_coverage_wrong': mw['predicted_coverage'],
            'mean_presence_prob_correct': float(presence_c.mean()),
            'mean_presence_prob_wrong': float(presence_w.mean()),
        })
        if number % 10 == 0:
            print(f'Processed {number}/{len(selected)} expressions', flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_expression.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    mean = lambda key: float(np.mean([r[key] for r in rows if r[key] is not None]))
    summary = {
        'n_expressions': len(rows),
        'n_same_video_negatives': sum(r['negative_type'] == 'same_video' for r in rows),
        'J_correct_mean': mean('J_correct'),
        'J_wrong_mean': mean('J_wrong'),
        'J_gain_correct_vs_wrong_mean': mean('J_gain_correct_vs_wrong'),
        'changed_mask_fraction_mean': mean('changed_mask_fraction'),
        'probability_change_mean': mean('mean_abs_probability_change'),
        'J_correct_present_mean': mean('J_correct_present'),
        'J_wrong_present_mean': mean('J_wrong_present'),
        'absent_target_frames_total': sum(r['gt_absent_frames'] for r in rows),
        'false_positive_frames_correct_total': sum(r['false_positive_frames_correct'] for r in rows),
        'presence_threshold': args.presence_threshold,
        'note': ('Same seed and sampling as E2; same-video negatives can share targets. '
                 'E3 applies presence gating to masks. Probability change uses raw '
                 'segmentation probabilities without gating. Not official MeViS J&F.'),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
