"""Offline CLIP text embedding computation. Download only at this step."""
import argparse
import json
from pathlib import Path
import torch
from tqdm import trange
from .data import normal_text


def texts_for_splits(root, splits):
    out = set()
    for split in splits:
        with open(Path(root) / split / "meta_expressions.json", encoding="utf-8") as f:
            data = json.load(f)["videos"]
        for vid in data.values():
            for exp in vid["expressions"].values():
                out.add(normal_text(exp["exp"]))
    return sorted(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "valid_u"])
    parser.add_argument("--out", default="outputs/clip_text.pt")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    from transformers import CLIPModel, CLIPTokenizerFast
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizerFast.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    texts = texts_for_splits(args.data, args.splits)
    cache = {}
    for start in trange(0, len(texts), args.batch_size):
        batch = texts[start:start+args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features.float(), dim=-1).cpu()
        cache.update({text: feat for text, feat in zip(batch, features)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    print(f"Cached {len(cache)} CLIP embeddings ({features.shape[-1]} dims) -> {args.out}")


if __name__ == "__main__":
    main()
