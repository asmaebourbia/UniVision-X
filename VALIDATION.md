# Validation status — 14 October 2026

## Verified from submitted HPC terminal outputs

- E2 training and full-video local evaluation on MeViS v2 `valid_u` (907 expressions): mean J=0.230774.
- E3 SLURM training job completed successfully on the NVIDIA RTX 2080 Ti, with 12 epochs and selected checkpoint at epoch 6.
- E3 full-video evaluation on `valid_u` (907 expressions): local mean J=0.310270; mean Dice=0.394590.
- 100-expression E2/E3 text-swap diagnostics and expression-level paired comparison were reported.
- These outcomes are operator-provided terminal results, not independently rerun from checkpoints in this GitHub publication workspace.

## Verified while preparing this publication package

- `python -m pytest -q tests/`: **6 passed** using CPU with synthetic inputs.
- Python syntax compilation of the repository modules: success.
- README relative links and exclusion of private absolute HPC paths: checked.

## Not yet verified

- Re-running the exact GPU experiments from the publication ZIP without comparison against the final live HPC checkout.
- Official MeViS v2 J&F scoring using the organizers' reference evaluation implementation.
- Multiple random seeds, controlled component-wise ablations, or independent test-set results.
- Optional diffusion, SAM2, SSL, LoRA or Mean Teacher modules as validated scientific contributions.

**The repository is a documented portfolio research prototype, not a peer-reviewed, benchmark-certified method.**
