#!/usr/bin/env python3
"""Create real RGB / GT / E2 / E3 galleries from saved MeViS PNG predictions.

CPU only, no model checkpoint needed. Run from the repository root.

Example:
    python scripts/make_qualitative_comparisons.py \
        --data data/MeViS_release/valid_u \
        --e2 outputs/E2_eval_clean --e3 outputs/E3_eval_clean \
        --out outputs/qualitative_e2_e3

Outputs a selected set of paired comparison images and manifest.csv.
These are qualitative illustrations; frame selection is disclosed in the manifest.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Reuse the tested dataset ground-truth decoding from the E2 diagnostic.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_univision_e2 import mask_for_frame

EXAMPLES = [
    ('4a9c4d01922f', '29'),  # Person feeding pandas
    ('4c18b41f0691', '5'),   # Dog and monkey
    ('ee6855dcefee', '7'),   # Rabbits
    ('16ae95660b3a', '20'), # Bear following elephant
    ('a1dbd8258387', '22'), # Bird eating/drinking
]


def load_scores(folder):
    csv_path = Path(folder) / 'per_expression.csv'
    with csv_path.open(newline='', encoding='utf-8') as f:
        return {(r['video'], str(r['exp_id'])): r for r in csv.DictReader(f)}


def read_pred(folder, vid, exp_id, frame, shape):
    path = Path(folder) / 'Annotations' / vid / exp_id / f'{frame}.png'
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved mask: {path}. Re-run evaluation with --save-png.')
    with Image.open(path) as im:
        return np.asarray(im.convert('L').resize(shape, Image.Resampling.NEAREST)) > 127


def overlay(image, mask, rgb_color, alpha=.48):
    x = np.asarray(image.convert('RGB')).copy()
    if mask.any():
        x[mask] = np.clip((1-alpha)*x[mask] + alpha*np.asarray(rgb_color), 0, 255).astype(np.uint8)
    return Image.fromarray(x)


def iou(pred, gt):
    union = (pred | gt).sum()
    return float((pred & gt).sum() / union) if union else 1.0


def wrap_text(draw, text, max_width, font):
    words = text.split()
    lines = []
    line = ''
    for word in words:
        nxt = word if not line else f'{line} {word}'
        if draw.textbbox((0,0), nxt, font=font)[2] > max_width and line:
            lines.append(line)
            line = word
        else:
            line = nxt
    if line:
        lines.append(line)
    return lines


def frame_choice(frames, video_dir, annotations, masks, e2, e3, vid, exp_id):
    """Prefer central frame with annotated target. Never choose frames using model scores."""
    n = len(frames)
    candidates = sorted(range(n), key=lambda x: abs(x - (n-1)/2))
    fallback = None
    for idx in candidates:
        frame = frames[idx]
        image_path = video_dir / f'{frame}.jpg'
        e2_path = Path(e2)/'Annotations'/vid/exp_id/f'{frame}.png'
        e3_path = Path(e3)/'Annotations'/vid/exp_id/f'{frame}.png'
        if not all(p.is_file() for p in (image_path,e2_path,e3_path)):
            continue
        with Image.open(image_path) as im:
            w,h = im.size
        gt = mask_for_frame(annotations, masks, idx, h, w)
        if fallback is None:
            fallback = (idx, frame, gt)
        if gt.any():
            return (idx, frame, gt), 'central-nearest target-present frame'
    if fallback is None:
        raise FileNotFoundError(f'No complete RGB+E2+E3 frame found for {vid}/{exp_id}')
    return fallback, 'central frame; target absent'


def make_one(vid, exp_id, data, meta, masks, e2, e3, out, scores_e2, scores_e3):
    rec = meta[vid]
    exp = rec['expressions'][exp_id]
    frames = sorted(rec['frames'])
    info, reason = frame_choice(frames, data/'JPEGImages'/vid,
                                exp.get('anno_id') or [], masks,
                                e2, e3, vid, exp_id)
    idx, frame, gt = info
    with Image.open(data/'JPEGImages'/vid/f'{frame}.jpg') as im:
        rgb = im.convert('RGB')
    p2 = read_pred(e2, vid, exp_id, frame, rgb.size)
    p3 = read_pred(e3, vid, exp_id, frame, rgb.size)
    j2, j3 = iou(p2,gt), iou(p3,gt)
    parts = [
        ('Original RGB', rgb),
        ('Ground truth', overlay(rgb, gt, (46,196,89))),
        (f'E2 mask  |  frame IoU {j2:.3f}', overlay(rgb, p2, (230,82,68))),
        (f'E3 mask  |  frame IoU {j3:.3f}', overlay(rgb, p3, (45,111,238))),
    ]
    panel_w, panel_h, margin = 355, 252, 14
    header, footer = 40, 92
    image = Image.new('RGB', (4*panel_w, panel_h+header+footer), 'white')
    dr = ImageDraw.Draw(image)
    for pi,(label,part) in enumerate(parts):
        shown=ImageOps.contain(part,(panel_w-2*margin,panel_h-2*margin))
        x=pi*panel_w+(panel_w-shown.width)//2
        y=header+(panel_h-shown.height)//2
        image.paste(shown,(x,y))
        dr.text((pi*panel_w+margin, 12), label, fill='#202020')
    dr.line((0,header+panel_h,4*panel_w,header+panel_h), fill='#d8d8d8',width=1)
    dr.text((12,header+panel_h+9),f'Video {vid} | expression {exp_id} | frame {frame}',fill='#444444')
    lines = wrap_text(dr, 'Expression: '+exp['exp'], 4*panel_w-30, ImageFont.load_default())[:2]
    for line_i,line in enumerate(lines):
        dr.text((12,header+panel_h+32+line_i*14),line,fill='#222222')
    dr.text((12,header+panel_h+72),'Red = E2 | Blue = E3 | Green = GT | Frame-level IoU shown above',fill='#555555')
    dest=out/f'{vid}_exp{exp_id}_E2_E3.png'
    image.save(dest,optimize=True)
    return {
      'image':str(dest), 'video':vid, 'exp_id':exp_id, 'frame':frame,
      'frame_selection':reason,'text':exp['exp'],
      'target_present':bool(gt.any()),'frame_J_E2':round(j2,6),'frame_J_E3':round(j3,6),
      'expression_J_E2':scores_e2[(vid,exp_id)]['J'],
      'expression_J_E3':scores_e3[(vid,exp_id)]['J']
    }


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',type=Path,default=Path('data/MeViS_release/valid_u'))
    p.add_argument('--e2',type=Path,default=Path('outputs/E2_eval_clean'))
    p.add_argument('--e3',type=Path,default=Path('outputs/E3_eval_clean'))
    p.add_argument('--out',type=Path,default=Path('outputs/qualitative_e2_e3'))
    p.add_argument('--cases',nargs='*',help='Optional video:expression pairs, e.g. 4a9c4d01922f:29')
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.data/'meta_expressions.json').read_text(encoding='utf-8'))['videos']
    masks=json.loads((args.data/'mask_dict.json').read_text(encoding='utf-8'))
    scores_e2=load_scores(args.e2)
    scores_e3=load_scores(args.e3)
    cases=[tuple(v.split(':',1)) for v in args.cases] if args.cases else EXAMPLES
    results=[]
    for vid,exp_id in cases:
        if (vid,exp_id) not in scores_e2 or (vid,exp_id) not in scores_e3:
            print(f'SKIP: expression not present in both CSVs: {vid}/{exp_id}')
            continue
        try:
            row=make_one(vid,exp_id,args.data,meta,masks,args.e2,args.e3,args.out,scores_e2,scores_e3)
            results.append(row)
            print(f"Saved {row['image']}: J(E2)={row['frame_J_E2']:.3f} J(E3)={row['frame_J_E3']:.3f}")
        except Exception as exc:
            print(f'SKIP: {vid}/{exp_id}: {exc}')
    if results:
        with (args.out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(f'Finished: {len(results)} paired examples in {args.out}.')
    if not results:
        raise SystemExit('No figures created; verify the PNG predictions and dataset paths.')

if __name__ == '__main__':
    main()
