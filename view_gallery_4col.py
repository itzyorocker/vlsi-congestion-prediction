"""
Extended gallery for the BEST model (unet_patch128_v2.pt).
Shows 4 columns: macro_region, input RUDY, true congestion, predicted.
18 samples total, split across 3 images (6 rows each) for readability.
"""
import torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from train_patch128_v2 import PatchCircuitNet, ResUNetSE, DEVICE, IN_FEATS

MODEL_FILE = "unet_patch128_v2.pt"
TOTAL_SAMPLES = 18
ROWS_PER_IMAGE = 6

ds = PatchCircuitNet()
print(f"dataset has {len(ds)} patches total")

net = ResUNetSE(cin=len(IN_FEATS)).to(DEVICE)
net.load_state_dict(torch.load(Path.home() / f"vlsi-project/{MODEL_FILE}",
                               map_location=DEVICE, weights_only=True))
net.eval()

step = max(1, len(ds) // TOTAL_SAMPLES)
indices = [i * step for i in range(TOTAL_SAMPLES)]

num_images = (TOTAL_SAMPLES + ROWS_PER_IMAGE - 1) // ROWS_PER_IMAGE

for img_idx in range(num_images):
    batch_indices = indices[img_idx * ROWS_PER_IMAGE : (img_idx + 1) * ROWS_PER_IMAGE]
    n = len(batch_indices)
    if n == 0:
        continue

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]

    titles = ["macro_region", "input RUDY", "true congestion", "predicted"]

    for row, i in enumerate(batch_indices):
        x, y = ds[i]
        with torch.no_grad():
            pred = net(x[None].to(DEVICE)).cpu()[0, 0]

        axes[row, 0].imshow(x[0], cmap="viridis")   # macro_region (index 0)
        axes[row, 1].imshow(x[1], cmap="viridis")   # RUDY (index 1)
        axes[row, 2].imshow(y[0], cmap="viridis")    # true congestion
        axes[row, 3].imshow(pred, cmap="viridis")    # predicted

        for col in range(4):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col])

    plt.tight_layout()
    out = Path.home() / f"vlsi-project/gallery_4col_part{img_idx+1}.png"
    plt.savefig(out, dpi=110)
    print("saved", out)

print(f"\nDone. Saved {num_images} images, {TOTAL_SAMPLES} total patches, 4 columns each.")	
