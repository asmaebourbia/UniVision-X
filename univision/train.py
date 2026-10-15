"""Training: supervised, optional frozen FastViT LoRA, Mean-Teacher consistency,
SAM2 teacher mask distillation. No use of validation masks for training.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .data import MeViS
from .model import UniVisionX
from .metrics import segmentation_loss
from .utils import seed_everything, available_device, ensure_dir, num_parameters, write_json, worker_seed


def make_model(args, pretrained=True):
    return UniVisionX(encoder=args.encoder, pretrained=pretrained,
                      width=args.width, use_text=not args.visual_only,
                      use_temporal=not args.no_temporal,
                      freeze_encoder=args.freeze_encoder,
                      lora_rank=args.lora_rank)


@torch.no_grad()
def update_ema(student, ema, decay=0.99):
    s = dict(student.named_parameters())
    for name, param in ema.named_parameters():
        param.lerp_(s[name].detach(), 1 - decay)
    sb = dict(student.named_buffers())
    for name, buf in ema.named_buffers():
        if buf.is_floating_point():
            buf.lerp_(sb[name].detach(), 1 - decay)
        else:
            buf.copy_(sb[name])


def train_one_epoch(model, ema, loader, opt, scaler, args, device):
    model.train()
    if ema is not None:
        ema.eval()
    total_loss = 0.0
    count = 0
    for batch in tqdm(loader, desc="train", leave=False):
        imgs = batch["video"].to(device)
        clean = batch["clean_video"].to(device)
        mask = batch["mask"].to(device)
        txt = batch["text_emb"].to(device)
        use_amp = args.amp and device.type == "cuda"
        labeled = torch.rand(imgs.shape[0], device=device) >= args.unlabeled_fraction
        if not labeled.any() and args.unsup_weight == 0:
            continue
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(imgs, txt)
            loss = logits.sum() * 0
            if labeled.any():
                loss = loss + segmentation_loss(logits[labeled], mask[labeled])
            if ema is not None and (~labeled).any() and args.unsup_weight > 0:
                with torch.no_grad():
                    weak_logits, _ = ema(clean[~labeled], txt[~labeled])
                    prob = weak_logits.sigmoid().float()
                    valid = ((prob >= args.pseudo_threshold) | (prob <= 1-args.pseudo_threshold))
                    pseudo = (prob >= 0.5).float()
                if valid.any():
                    perpx = F.binary_cross_entropy_with_logits(logits[~labeled].float(), pseudo, reduction="none")
                    unsup = (perpx * valid).sum() / valid.sum().clamp_min(1)
                    loss = loss + args.unsup_weight * unsup
            if args.teacher_weight > 0:
                teacher_valid = batch["teacher_valid"].to(device)  # B,T
                if teacher_valid.any():
                    teacher = batch["teacher_mask"].to(device)
                    pred_selected = logits[teacher_valid]
                    target_selected = teacher[teacher_valid]
                    loss = loss + args.teacher_weight * F.binary_cross_entropy_with_logits(pred_selected, target_selected)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            update_ema(model, ema, args.ema_decay)
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(1, count)


@torch.inference_mode()
def validate(model, loader, device, amp=False):
    model.eval()
    total_intersection = 0
    total_union = 0
    total_dice_num = 0
    total_dice_den = 0
    for batch in tqdm(loader, desc="validation", leave=False):
        imgs = batch["video"].to(device)
        target = batch["mask"].to(device).bool()
        txt = batch["text_emb"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits, _ = model(imgs, txt)
        pred = logits.sigmoid() >= 0.5
        total_intersection += int((pred & target).sum().item())
        total_union += int((pred | target).sum().item())
        total_dice_num += 2 * int((pred & target).sum().item())
        total_dice_den += int(pred.sum().item() + target.sum().item())
    return {
        "pixel_micro_iou": total_intersection / max(1, total_union),
        "pixel_micro_dice": total_dice_num / max(1, total_dice_den),
    }


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="outputs/experiment")
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--visual-only", action="store_true")
    ap.add_argument("--encoder", default="fastvit_t8.apple_in1k")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--no-temporal", action="store_true")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--clip-length", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--corruption-probability", type=float, default=0.7)
    ap.add_argument("--val-corruption", choices=["clean","blur","noise","fog","dark"], default="clean")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--teacher-mask-root", default=None)
    ap.add_argument("--teacher-weight", type=float, default=0)
    ap.add_argument("--unlabeled-fraction", type=float, default=0)
    ap.add_argument("--unsup-weight", type=float, default=0)
    ap.add_argument("--pseudo-threshold", type=float, default=0.9)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--init-encoder", default=None, help="Load SSL encoder state from pretrain_ssl.py")
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main():
    args = get_args()
    if not 0 <= args.unlabeled_fraction < 1:
        raise ValueError("unlabeled-fraction must be [0, 1)")
    if args.unlabeled_fraction > 0 and args.unsup_weight == 0:
        raise ValueError("Use --unsup-weight > 0 when withholding annotations")
    if args.teacher_weight > 0 and not args.teacher_mask_root:
        raise ValueError("--teacher-weight > 0 requires --teacher-mask-root")
    seed_everything(args.seed)
    ensure_dir(args.out)
    write_json(vars(args), Path(args.out) / "config.json")
    device = available_device()
    tr = MeViS(args.data, "train", args.image_size, args.clip_length, train=True,
               corruption_probability=args.corruption_probability,
               text_cache=args.text_cache, visual_only=args.visual_only,
               teacher_mask_root=args.teacher_mask_root, limit=args.limit_train)
    va = MeViS(args.data, "valid_u", args.image_size, args.clip_length, train=False,
               corrupt_mode=args.val_corruption, text_cache=args.text_cache,
               visual_only=args.visual_only, limit=args.limit_val)
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=(device.type=="cuda"), worker_init_fn=worker_seed)
    vloader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=(device.type=="cuda"))
    model = make_model(args, pretrained=not args.no_pretrained).to(device)
    if args.init_encoder:
        state = torch.load(args.init_encoder, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state["encoder"] if "encoder" in state else state, strict=True)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = copy.deepcopy(model).eval() if args.unsup_weight > 0 else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    start_epoch = 0
    best = -1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        if ema is not None and state.get("ema") is not None:
            ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"] + 1
        best = state.get("best", -1)
    print("device:", device, "params(total,trainable):", num_parameters(model), "samples(train,val):", len(tr), len(va))
    history = []
    for epoch in range(start_epoch, args.epochs):
        start = time.time()
        loss = train_one_epoch(model, ema, loader, opt, scaler, args, device)
        metrics = validate(model, vloader, device, args.amp)
        score = metrics["pixel_micro_iou"]
        result = {"epoch": epoch+1, "train_loss": loss, **metrics, "seconds": round(time.time()-start, 2)}
        history.append(result)
        print(json.dumps(result))
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "ema": ema.state_dict() if ema else None, "best": max(best, score), "config": vars(args)}
        torch.save(ckpt, Path(args.out) / "last.pt")
        if score > best:
            best = score
            torch.save(ckpt, Path(args.out) / "best.pt")
        write_json(history, Path(args.out) / "history.json")
    print("Training finished. For full MeViS evaluation run: python -m univision.evaluate ...")


if __name__ == "__main__":
    main()
