import argparse
import time
import torch
from .evaluate import build_checkpoint_model
from .utils import available_device, num_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    device = available_device()
    model, cfg = build_checkpoint_model(a.checkpoint, device)
    im = torch.rand((1, a.frames, 3, a.height, a.width), device=device)
    emb = torch.randn((1, 512), device=device)
    def sync():
        if device.type == "cuda": torch.cuda.synchronize()
    with torch.inference_mode():
        for i in range(a.warmup):
            model(im, emb)
        sync()
        st = time.perf_counter()
        for i in range(a.iters):
            model(im, emb)
        sync()
    sec = (time.perf_counter()-st)/a.iters
    print({"device":str(device),"parameters":num_parameters(model),
           "seconds_per_clip":round(sec,4),"fps_including_encoder":round(a.frames/sec,2),
           "image_size":[a.height,a.width],"batch_size":1})
    print("For benchmark comparison use same GPU, batch size, resolution and timing conditions.")


if __name__ == "__main__":
    main()
