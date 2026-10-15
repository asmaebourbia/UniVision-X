#!/usr/bin/env python3
"""Analyze UniVision-X E2 text-swap diagnostic without GPU.

The swapped description is matched against metadata of the *same* video,
which lets us flag pairs likely referring to the same annotated objects.
Bootstrap resamples expressions, not individual frames.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def normalize(s):
    return ' '.join(str(s).lower().strip().split())


def get_ids(exp):
    return set(map(str, exp.get('anno_id') or []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='outputs/E2_text_diagnostic/per_expression.csv')
    p.add_argument('--meta', default='data/MeViS_release/valid_u/meta_expressions.json')
    p.add_argument('--out', default='outputs/E2_text_diagnostic/paired_analysis.json')
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--bootstrap', type=int, default=3000)
    a = p.parse_args()

    with open(a.csv, encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    with open(a.meta, encoding='utf-8') as f:
        videos = json.load(f)['videos']
    if not rows:
        raise ValueError('CSV vide')

    def lookup_group(r):
        if r.get('negative_type') != 'same_video':
            return 'other_video'
        vi = videos.get(r['video'])
        if not vi:
            return 'video_not_found'
        expressions = vi['expressions']
        gt_exp = expressions.get(str(r['exp_id']))
        wrongs = [e for e in expressions.values() if normalize(e.get('exp', '')) == normalize(r['wrong_text'])]
        if gt_exp is None or len(wrongs) != 1:
            return 'ambiguous_text_match'
        gt_ids = get_ids(gt_exp)
        wrong_ids = get_ids(wrongs[0])
        if not gt_ids or not wrong_ids:
            return 'empty_target_in_pair'
        if gt_ids & wrong_ids:
            return 'shared_annotation_ids'
        return 'disjoint_annotation_ids'

    for r in rows:
        r['_group'] = lookup_group(r)
        for k in ('J_correct', 'J_wrong', 'J_gain_correct_vs_wrong',
                  'changed_mask_fraction', 'mean_abs_probability_change',
                  'gt_absent_frames', 'false_positive_frames_correct', 'frames',
                  'pred_coverage_correct', 'pred_coverage_wrong'):
            r[k] = float(r[k])

    rng = np.random.default_rng(a.seed)
    def stats(items):
        if not items:
            return None
        differences = np.array([x['J_correct'] - x['J_wrong'] for x in items])
        means = rng.choice(differences, size=(a.bootstrap, len(items)), replace=True).mean(1)
        absent = sum(x['gt_absent_frames'] for x in items)
        false = sum(x['false_positive_frames_correct'] for x in items)
        return {
            'n_expressions': len(items),
            'J_correct_mean': float(np.mean([x['J_correct'] for x in items])),
            'J_wrong_mean': float(np.mean([x['J_wrong'] for x in items])),
            'J_margin_mean': float(differences.mean()),
            'J_margin_median': float(np.median(differences)),
            'J_margin_95pct_bootstrap_CI': [float(x) for x in np.percentile(means, [2.5, 97.5])],
            'fraction_margin_positive': float((differences > 0).mean()),
            'fraction_masks_changed': float(np.mean([x['changed_mask_fraction'] for x in items])),
            'mean_abs_probability_change': float(np.mean([x['mean_abs_probability_change'] for x in items])),
            'absent_target_frames': int(absent),
            'false_positive_absent_frames': int(false),
            'false_positive_frame_rate_when_absent': float(false / absent) if absent else None,
        }

    groups = defaultdict(list)
    for row in rows:
        groups[row['_group']].append(row)
    report = {
        'all': stats(rows),
        'groups': {k: stats(v) for k, v in sorted(groups.items())},
        'group_counts': dict(Counter(x['_group'] for x in rows)),
        'notes': [
            'Les paires avec ids disjoints sont des negatives plus fortes mais pas une preuve parfaite de cibles differentes dans chaque frame.',
            'L intervalle bootstrap est conditionnel a cet echantillon de 100 expressions et depend des correlations inter-expressions.',
            'Le taux de faux positifs compte une frame comme positive meme pour un seul pixel predit positif.',
            'Ni la marge J ni le taux de faux positifs ne sont des scores officiels MeViS.',
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
