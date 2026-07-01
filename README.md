# VLSI Routing Congestion Prediction

Deep learning model for predicting post-routing congestion maps from 
placement-stage chip design features, using a Residual U-Net with 
Squeeze-and-Excitation (SE) attention blocks, trained on the 
CircuitNet-N14 dataset.

## Result

**Best model: Pearson correlation 0.3107**
(128x128 native-resolution patches, 7 input channels)

| Approach | Pearson |
|---|---|
| Naive U-Net, plain MSE | 0.0692 |
| + biased loss (sparsity fix) | 0.1488 |
| ResUNet+SE, full image | 0.1957 |
| Patch-based, 128x128, 3 channels | 0.2045 |
| **Patch-based, 128x128, 7 channels** | **0.3107** |

## Architecture

- Residual U-Net encoder-decoder with skip connections
- Squeeze-and-Excitation (SE) blocks for channel attention
- GroupNorm (switched from BatchNorm after diagnosing a training 
  collapse on small batch sizes with sparse data)
- Input: 7 channels (macro_region, RUDY, RUDY_pin, RUDY_long, 
  RUDY_short, RUDY_pin_long, cell_density)
- Target: GR horizontal + vertical overflow congestion maps

## Key findings

- **Patch size matters**: swept 256 -> 128 -> 64px patches. 128px was 
  the sweet spot — large enough for context, small enough to force 
  local learning instead of tracing macro-block outlines.
- **Channel expansion was the biggest single lever**: going from 3 to 
  7 input channels improved Pearson by ~52%, more than the entire 
  patch-size sweep combined.
- **Diagnosed a silent training failure**: a deeper model's loss froze 
  at an identical value for 20 epochs. Traced to BatchNorm instability 
  on small batches of sparse data — fixed by switching to GroupNorm.
- **Caught a misleading result**: an overlapping-patch experiment 
  produced a visually richer prediction gallery but scored LOWER on 
  Pearson than the standard model — diagnosed as train/validation 
  leakage from near-duplicate overlapping crops, not a real 
  generalization gain.
- Learned to distrust SSIM alone as a metric — it stayed artificially 
  high (~0.98) throughout, inflated by how much of each chip is empty 
  space. Pearson correlation was the more honest signal throughout.

## Dataset

[CircuitNet-N14](https://circuitnet.github.io/) — three chip designs 
(Vortex-small, nvdla-small, zero-riscy), routability features 
(RUDY variants, macro region, cell density) and post-routing 
congestion ground truth.

## Files

- `train.py` / `train_patch*.py` — training scripts for each model variant
- `eval*.py` — evaluation scripts (NRMS, SSIM, Pearson)
- `view_patch*.py`, `gallery.py` — visualization scripts
- `PROJECT_NOTES.md` — full experiment log and lessons learned
- `images/` — result galleries showing prediction quality progression# vlsi-congestion-prediction
Deep learning model (Residual U-Net + SE) for predicting VLSI routing congestion from placement-stage features, trained on CircuitNet-N14
