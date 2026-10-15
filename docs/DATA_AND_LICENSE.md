# Data sources, visual figures and license considerations

## MeViS source

- Dataset: **MeViS / MeViS v2**, from the original authors, official project repository: https://github.com/henghuiding/MeViS .
- The original dataset reports **CC BY-NC-SA 4.0** terms and is intended for **non-commercial research**. Confirm the latest authoritative terms before publishing or redistributing any derivative material.
- **No original MeViS dataset archives, annotations, videos, precomputed features or model weights** are bundled in this repository.

## What the repository *does* include

`assets/qualitative/` contains **three real visualization composites** created from MeViS frames and annotation masks, overlaid with **UniVision-X E2 and E3 predictions**:

| Figure | Video and expression | Intended placement |
|---|---|---|
| `rabbit_e2_vs_e3.png` | `ee6855dcefee`, expression 7 | Main result: E3 better localizes the referred rabbit |
| `dog_e2_vs_e3.png` | `4c18b41f0691`, expression 5 | Main result: modest improvement |
| `panda_failure_e2_vs_e3.png` | `4a9c4d01922f`, expression 29 | Collapsed failure case; do not advertise as a successful result |

These composites **contain adaptations of underlying dataset frames**. They are not public-domain or unrestricted stock photography. **Credit to:** MeViS dataset authors (Ding et al.), with UniVision-X segmentation overlays added for analysis. Redistribution/reuse must honor the dataset's **attribution, non-commercial and share-alike requirements** where applicable. A permissive code license does **not** relicense these images.

Frame-level numerical values come from `results/qualitative_manifest_selected.csv`. Original additional images (other frames) and full prediction masks are not needed in the public repository.

## Code and external model licenses

- No project source-code `LICENSE` is inserted by default: verify rights and ownership of source code, including any work done under an institutional contract, before licensing it.
- Pretrained components such as **FastViT**, **CLIP**, `timm`, Transformers and PyTorch have separate licenses and terms; consult the original repositories and do not redistribute their weights without checking permissions.
- Exclude personally identifying HPC logs, local dataset paths, API keys, `.env` files, large checkpoints and annotations when publishing.

## Before a public push

1. Verify dataset redistribution rights for the three composite figures.
2. Check potential institutional or collaborator IP requirements for the code.
3. Check that exported masks, images and training logs are not unintentionally included.
4. Only add a repository-wide license if it is appropriate for your original code; clarify that third-party image assets are separately licensed.
