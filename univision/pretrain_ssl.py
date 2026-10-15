"""Optional genuine self-supervised SimCLR-style pretraining for FastViT.
Two corruption views of the SAME source frame; all other batch frames are negatives.
Works without segmentation labels and without text embeddings.
"""
import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from .corruptions import random_corruption
from .utils import seed_everything, available_device, ensure_dir


class UnlabeledMeViSFrames(Dataset):
    def __init__(self, root, split="train", size=224, limit=0):
        root = Path(root) / split
        with open(root / "meta_expressions.json", encoding="utf-8") as f:
            meta = json.load(f)["videos"]
        self.frames = []
        for vid, record in sorted(meta.items()):
            for name in record["frames"]:
                self.frames.append(root / "JPEGImages" / vid / f"{name}.jpg")
        if limit:
            self.frames = self.frames[:limit]
        self.size = size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        with Image.open(self.frames[idx]) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.
        # Two DIFFERENT views but same geometric content, preserving identity.
        a = random_corruption(x.unsqueeze(0), probability=1.0).squeeze(0)
        b = random_corruption(x.unsqueeze(0), probability=1.0).squeeze(0)
        return a, b


class ContrastiveNetwork(nn.Module):
    def __init__(self, name, pretrained, dim=128):
        super().__init__()
        import timm
        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True)
        ch = self.encoder.feature_info.channels()[-1]
        self.proj = nn.Sequential(nn.Linear(ch, ch), nn.ReLU(), nn.Linear(ch, dim))
        self.register_buffer("mean", torch.tensor([.485,.456,.406])[None,:,None,None], persistent=False)
        self.register_buffer("std", torch.tensor([.229,.224,.225])[None,:,None,None], persistent=False)

    def forward(self, x):
        z = self.encoder((x-self.mean)/self.std)[-1].mean((-1,-2))
        return F.normalize(self.proj(z), dim=-1)


def nt_xent(a, b, temperature=.2):
    if a.size(0) < 2:
        raise ValueError("Contrastive pretraining needs batch_size >= 2")
    z = torch.cat([a,b], 0)
    logits = z @ z.T / temperature
    diag = torch.eye(z.size(0), device=z.device).bool()
    logits = logits.masked_fill(diag, float("-inf"))
    labels = torch.cat([torch.arange(len(a), 2*len(a), device=z.device),
                        torch.arange(0, len(a), device=z.device)])
    return F.cross_entropy(logits, labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="outputs/ssl_fastvit")
    p.add_argument("--encoder", default="fastvit_t8.apple_in1k")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--no-pretrained", action="store_true")
    a = p.parse_args()
    seed_everything(a.seed)
    ensure_dir(a.out)
    device = available_device()
    ds = UnlabeledMeViSFrames(a.data, size=a.image_size, limit=a.limit)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                    drop_last=True)
    net = ContrastiveNetwork(a.encoder, pretrained=not a.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr)
    for epoch in range(a.epochs):
        net.train()
        avg = 0.0
        for x,y in tqdm(dl, desc=f"SSL epoch {epoch+1}", leave=False):
            x,y = x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            # All images are jointly forwarded: BN sees both views.
            z = net(torch.cat([x,y], 0))
            loss = nt_xent(*z.chunk(2, dim=0))
            loss.backward()
            optimizer.step()
            avg += loss.item()
        print(f"SSL epoch {epoch+1}: loss={avg/max(1,len(dl)):.4f}")
        torch.save({"encoder": net.encoder.state_dict()}, Path(a.out)/"encoder_ssl.pt")
    print("Weights ready for supervised --init-encoder; compare against ImageNet-only weights.")


if __name__ == "__main__":
    main()
