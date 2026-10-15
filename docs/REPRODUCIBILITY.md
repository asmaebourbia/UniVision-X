# Reproducibility guide

The project is intended for Linux + CUDA and, optionally, SLURM. Model weights, external dataset files and downloaded CLIP/FastViT weights are excluded from GitHub.

## Runtime used in the reported experiments

- GPU: NVIDIA GeForce RTX 2080 Ti, 11 GB
- Python 3.11.7
- `torch==2.7.0+cu118`, `torchvision==0.22.0+cu118`
- `timm==1.0.30`, `transformers==4.57.6`
- Train/val: MeViS v2 `train`/`valid_u` (27,502/907 expressions)

Install a GPU-compatible PyTorch wheel separately, then `pip install -r requirements.txt`. Run tests with `python -m pytest -q tests/`.

## Check data structure

```bash
ls data/MeViS_release/train/JPEGImages
ls data/MeViS_release/train/mask_dict.json
ls data/MeViS_release/valid_u/meta_expressions.json
```

If `meta_expressions_v2.json` exists without `meta_expressions.json`, create an appropriate symlink (do not rename/rewrite content without need).

## Run without pretrained weights to check the pipeline

```bash
python -m univision.synthetic --out data/synthetic --image-size 96
python -m pytest -q tests
python -m univision.train \
  --data data/synthetic --text-cache data/synthetic/SYNTHETIC_TEXT_ONLY.pt \
  --encoder toy --no-pretrained --epochs 1 --image-size 64 \
  --clip-length 2 --batch-size 2 --workers 0 --width 16 \
  --limit-train 4 --limit-val 2 --out outputs/SMOKE_TEST
```

Toy encoder/synthetic text is an integration check, **not a research baseline**.

## GPU runs (from repository root)

```bash
mkdir -p logs outputs
sbatch scripts/e3_preflight_2080ti.sbatch
squeue -u "$USER"
# Check the preflight job's success with sacct or slurm output
sbatch scripts/e3_train_2080ti.sbatch
# Wait for training completion before evaluating
sbatch scripts/e3_eval_2080ti.sbatch
```

SLURM submission assumes current directory is repository root via `$SLURM_SUBMIT_DIR`. For non-standard setups, set `UNIVISION_PROJECT_DIR=/path/to/UniVision-X` when submitting and customize the SLURM partition/GRES/QoS/time. Avoid running the same long job twice.

## Direct evaluation (without SLURM)

```bash
python -m univision.e3_evaluate \
  --checkpoint outputs/E3_fastvit_t8_2080ti/best.pt \
  --token-cache outputs/clip_tokens_32.pt \
  --data data/MeViS_release --split valid_u \
  --chunk-size 2 --out outputs/E3_eval_clean
```

Result files: `outputs/E3_eval_clean/summary.json`, `per_expression.csv`, and original-resolution PNG mask predictions in `Annotations/`.

Compare local per-expression J scores only when both experiment outputs share the same dataset split and evaluation convention. To compute the official MeViS J&F, follow the dataset authors' official `tools/eval_mevis.py` script and adjust its expected output directory; **do not label `boundary_F_proxy` as official**.

## Model limitations and checkpoints

`best.pt` and `last.pt` were generated on the user's HPC server but are not part of this distribution. To publish a trained model, verify institution rights, pretrained weight redistribution restrictions, and size limits. Prefer a release artifact with model card and checksum instead of directly committing a 16 MB checkpoint to the code repository.

**Note:** the contents of this GitHub package were assembled from the previously shared E2 archive and E3 source patch, not downloaded directly from the HPC checkout. If any source files were manually edited after the GPU experiments, inspect and synchronize them before claiming exact run-to-source provenance.
