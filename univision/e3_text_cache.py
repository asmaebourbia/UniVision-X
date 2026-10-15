"""Create WORD-LEVEL CLIP cache (not the E2 pooled 512D cache)."""
import argparse
from pathlib import Path
import torch
from tqdm.auto import tqdm
from .text_cache import texts_for_splits


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "valid_u"])
    p.add_argument("--out", default="outputs/clip_tokens_32.pt")
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    p.add_argument("--max-length", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    args=p.parse_args()
    if args.max_length < 4 or args.max_length > 77:
        raise ValueError("CLIP max-length must be in [4,77]")
    from transformers import CLIPModel, CLIPTokenizerFast
    tokenizer=CLIPTokenizerFast.from_pretrained(args.model)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=CLIPModel.from_pretrained(args.model).to(device).eval()
    text_model=model.text_model
    texts=texts_for_splits(args.data,args.splits)
    result={}
    truncated=0
    for start in tqdm(range(0,len(texts),args.batch_size),desc="CLIP word tokens"):
        batch=texts[start:start+args.batch_size]
        # Count descriptions that would have been truncated (for scientific reporting).
        full=tokenizer(batch, padding=False, truncation=False)
        truncated += sum(len(ids)>args.max_length for ids in full['input_ids'])
        tok=tokenizer(batch, padding="max_length", truncation=True,max_length=args.max_length,return_tensors="pt")
        inp={k:v.to(device) for k,v in tok.items() if k in ('input_ids','attention_mask','position_ids')}
        with torch.inference_mode():
            hs=text_model(**inp).last_hidden_state.float().cpu().half()
        mask=tok['attention_mask'].to(torch.uint8).cpu()
        for k,key in enumerate(batch):
            result[key]={"tokens":hs[k].contiguous(), "attention_mask":mask[k].contiguous()}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.out)
    print(f"Saved {len(result)} tokenized CLIP expressions to {args.out}; truncations={truncated} max_length={args.max_length}")

if __name__=="__main__":
    main()
