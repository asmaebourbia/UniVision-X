"""OPTIONAL diffusion image-to-image exploratory study only.

Generated images MAY alter geometry. Output images NEVER automatically inherit
segmentation masks, and are NEVER automatically added to MeViS training.
Require human verification or geometry-consistent image editing prior to use.
"""
import argparse
from pathlib import Path
from PIL import Image
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Single image for qualitative experiment")
    p.add_argument("--prompt", default="A real street scene on a rainy foggy day, photograph")
    p.add_argument("--out", default="outputs/generated_sample.png")
    p.add_argument("--model", default="stabilityai/sd-turbo")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Diffusion prototype requires GPU; run separately from training")
    from diffusers import AutoPipelineForImage2Image
    pipe = AutoPipelineForImage2Image.from_pretrained(a.model, torch_dtype=torch.float16, variant="fp16").to("cuda")
    image = Image.open(a.image).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
    with torch.inference_mode():
        out = pipe(prompt=a.prompt, image=image, strength=0.5,
                   guidance_scale=0.0, num_inference_steps=2).images[0]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.save(a.out)
    print("Qualitative diffusion output saved:", a.out)
    print("WARNING: Synthetic geometry may differ. Do not reuse original segmentation masks.")


if __name__ == "__main__":
    main()
