# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deep-learning project predicting post-routing congestion maps from placement-stage
VLSI features. Residual U-Net + Squeeze-and-Excitation attention, trained on
CircuitNet-N14 (three chip designs: Vortex-small, nvdla-small, zero-riscy).
See `README.md` for the current best result (Pearson 0.3107) and the experiment
log of what was tried and why.

## Environment

Use the `vlsi` conda environment — it has torch 2.5.1 (CUDA-enabled), numpy,
matplotlib. There is no requirements.txt; the env already exists.

```bash
conda activate vlsi
python train_patch128_v2.py        # e.g. run the current best training script
```

Scripts assume the repo lives at `~/vlsi-project` (many use `Path.home() /
"vlsi-project"` rather than relative paths) — run them from that checkout.

## Running things

There is no test suite, build step, or linter in this repo. The workflow is
entirely: run a training script -> run its matching eval script -> optionally
view results with a visualization script.

```bash
python train_patch128_v2.py        # trains, saves best checkpoint by val loss
python eval_patch128_v2.py         # loads that checkpoint, reports NRMS/SSIM/Pearson
python view_patch.py <ckpt.pt> <train_module_name>   # visualize predictions
```

Each `train_*.py` script is fully self-contained (dataset class, model,
training loop all in one file) and paired 1:1 with an `eval_*.py` of the same
suffix, which imports the dataset/model classes back out of the training
script rather than redefining them. When adding a new experiment variant,
follow this existing pattern: copy the nearest train script, give it a new
suffix, point its checkpoint save path at a new `.pt` filename, and write a
matching `eval_<suffix>.py` that imports from it.

Checkpoints (`unet*.pt`) and generated images (`*.png`) are committed to the
repo root as experiment artifacts, not gitignored — that's intentional here,
not an oversight.

## Architecture (the part that's shared across every script)

Every `train_*.py` / `eval_*.py` file redefines the same core pieces inline
(no shared module — this is copy-then-modify, by design, so each experiment
is a fully reproducible standalone artifact):

- **`ResUNetSE`**: encoder-decoder U-Net, 3 down/up stages, each stage is a
  `ResSEBlock` (residual conv block + Squeeze-Excitation channel attention).
- **GroupNorm, not BatchNorm** — this was a deliberate fix after BatchNorm
  caused a silent training collapse (loss frozen for 20 epochs) on small
  batches of sparse congestion maps. Do not swap this back to BatchNorm.
- **`loss_fn`**: weighted MSE, `w = 1 + WEIGHT * target` (WEIGHT=20) — a fix
  for label sparsity (most of a congestion map is near-zero; unweighted MSE
  collapses to predicting all zeros).
- **Data pipeline**: reads `.npz` feature/target maps from `data/<design>/...`
  (see below), min-max normalizes each input channel independently, and
  either resizes the whole chip to a fixed size (`train.py`, whole-image
  approach) or crops fixed-size patches on a grid (`train_patch*.py` family,
  the higher-performing approach). Patch scripts index `(paths, y0, x0)`
  tuples up front and crop lazily in `__getitem__`.
- **Evaluation metrics** (defined identically in each `eval_*.py`): Pearson
  correlation (the primary/most trustworthy metric), NRMS, and a hand-rolled
  windowed SSIM. Note from README: SSIM is misleadingly high (~0.98) here
  because most of every chip is empty space — trust Pearson over SSIM.
- **Train/val split**: `random_split` with `generator=...manual_seed(0)` — eval
  scripts reconstruct the identical split to evaluate only on held-out data.
  When writing a new eval script for an existing train script, copy the exact
  split logic (same seed, same fraction) or the "held-out" set won't match
  what training actually held out.

## Naming convention across script variants

The suffixes encode the experiment axis being varied — check a script's
docstring (each one has a short comment explaining what changed from the
previous variant) before assuming what a given file does:

- `train.py` — original whole-chip (resized to 384x384), 3 input channels.
- `train_patch.py` / `train_patch64.py` / `train_patch128*.py` — patch-based
  variants sweeping patch size (256/64/128px).
- `train_patch128_v2.py` — 128px patches, expanded to 7 input channels (the
  channel-count jump that drove most of the accuracy gain — see README).
- `train_patch128_v3.py` — same as v2 but with overlapping patches
  (stride=64 instead of 128). Produced a visually nicer gallery but a *worse*
  Pearson than v2 — diagnosed as train/val leakage from near-duplicate
  overlapping crops. Kept in the repo as a documented negative result, not
  a regression to fix.
- `train_holdout_design.py` — generalization test: trains on 2 of the 3
  chip designs and evaluates on the third, entirely unseen design (edit the
  `HOLDOUT` constant at the top to pick which). Different from the usual
  random split — tests cross-design generalization, not just held-out samples
  of the same designs.
- `eval_by_macro_density.py` — no retraining; buckets the existing
  patch128_v2 validation set by macro-cell density and reports Pearson per
  bucket, to test whether accuracy depends on how much of a patch is
  covered by macro blocks.

When adding a new variant, prefer this suffix-and-docstring convention over
introducing a config-flag/parameterized script — it's what keeps every past
experiment independently re-runnable from git history.

## Data layout

`data/<design>/<feature_group>/<feature_name>/<sample>.npz`, e.g.:

```
data/Vortex-small/RUDY/RUDY_pin/Vortex-small_freq_200_mp_1_..._fi_ap.npz
data/Vortex-small/macro_region/...npz
data/Vortex-small/cell_density/...npz
data/Vortex-small/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow/...npz
```

Each `.npz` has a single array (`z[z.files[0]]`). Sample filenames encode the
placement configuration (frequency, macro placement variant, utilization,
etc.) and are shared as the join key across every feature/target folder for
a given design — a training script builds its dataset index by listing one
target folder's filenames and checking the same name exists under every
required input feature folder. `data/*.tar.gz` are the original archives the
`data/<design>/` directories were extracted from.

The three designs have very different macro density (0% / ~33.5% / ~27% for
Vortex-small / nvdla-small / zero-riscy respectively) — relevant context if
touching anything related to generalization or per-design behavior.
