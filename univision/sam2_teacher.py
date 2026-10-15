"""OPTIONAL offline SAM 2 pseudo-mask generator for *training split only*.

Important: initializes SAM 2 from GT on the FIRST frame where the object is
present. This is a mask-prompted teacher, NOT a text-only RVOS baseline.
Do not use validation masks to generate pseudo-annotations for training!
"""
import argparse
import json
from pathlib import Path
from contextlib import nullcontext
import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm
from .data import decode_rle
from .utils import ensure_dir


def make_gt(r, mask_dict, idx, h, w):
    output = np.zeros((h,w), dtype=bool)
    for ann_id in r["ann_ids"]:
        annotations = mask_dict.get(ann_id, [])
        if idx < len(annotations) and annotations[idx] is not None:
            output |= decode_rle(annotations[idx]).astype(bool)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="train", choices=["train"])
    p.add_argument("--sam2-checkpoint", required=True, help="SAM 2.1 official .pt checkpoint")
    p.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    p.add_argument("--out", default="outputs/sam2_teacher/train")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This optional SAM 2 module expects a CUDA GPU")
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device="cuda")
    splitdir = Path(args.data) / args.split
    with open(splitdir / "meta_expressions.json", encoding="utf-8") as f:
        meta = json.load(f)["videos"]
    with open(splitdir / "mask_dict.json", encoding="utf-8") as f:
        mask_dict = json.load(f)
    records = []
    for vid, info in sorted(meta.items()):
        for eid, expr in sorted(info["expressions"].items()):
            records.append({"video":vid,"exp_id":str(eid),"frames":sorted(info["frames"]),
                            "ann_ids":[str(x) for x in (expr.get("anno_id") or [])]})
    if args.limit:
        records = records[:args.limit]
    stats = {"processed": 0, "skipped_empty": 0}
    for r in tqdm(records, desc="SAM2 pseudo-labels"):
        folder = splitdir / "JPEGImages" / r["video"]
        with Image.open(folder / (r["frames"][0] + ".jpg")) as first:
            w,h = first.size
        first_visible = None
        for i in range(len(r["frames"])):
            init_mask = make_gt(r, mask_dict, i, h, w)
            if init_mask.any():
                first_visible = i
                break
        if first_visible is None:
            stats["skipped_empty"] += 1
            continue
        output = Path(args.out) / r["video"] / r["exp_id"]
        ensure_dir(output)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state(video_path=str(folder), offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_mask(state, frame_idx=first_visible, obj_id=1, mask=init_mask)
            for frame_idx, ids, out_logits in predictor.propagate_in_video(state, start_frame_idx=first_visible):
                # The requested UNION target is a single SAM object (id=1).
                idx_obj = list(ids).index(1)
                pred = (out_logits[idx_obj, 0] > 0).cpu().numpy().astype(np.uint8) * 255
                Image.fromarray(pred).save(output / (r["frames"][frame_idx] + ".png"))
            predictor.reset_state(state)
        stats["processed"] += 1
    print("SAM2 teacher generation:", stats, "->", args.out)


if __name__ == "__main__":
    main()
