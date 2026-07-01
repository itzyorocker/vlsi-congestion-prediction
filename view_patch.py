"""
Visualize predictions from a patch-trained model.
Usage:  python view_patch.py unet_patch.pt train_patch       (256-patch model)
        python view_patch.py unet_patch128.pt train_patch128 (128-patch model)
"""
import sys, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import importlib

model_file = sys.argv[1] if len(sys.argv) > 1 else "unet_patch.pt"
module_name = sys.argv[2] if len(sys.argv) > 2 else "train_patch"

mod = importlib.import_module(module_name)
PatchCircuitNet = mod.PatchCircuitNet
ResUNetSE = mod.ResUNetSE
DEVICE = mod.DEVICE

ds = PatchCircuitNet()

net = ResUNetSE().to(DEVICE)
net.load_state_dict(torch.load(Path.home() / f"vlsi-project/{model_file}",
                               map_location=DEVICE, weights_only=True))
net.eval()

N = 6
fig, axes = plt.subplots(N, 3, figsize=(9, 3 * N))
step = max(1, len(ds) // N)

for row in range(N):
    i = row * step
    x, y = ds[i]
    with torch.no_grad():
        pred = net(x[None].to(DEVICE)).cpu()[0, 0]
    axes[row, 0].imshow(x[1], cmap="viridis")   # RUDY channel
    axes[row, 1].imshow(y[0], cmap="viridis")
    axes[row, 2].imshow(pred, cmap="viridis")
    for col, title in zip(range(3), ["input RUDY", "true congestion", "predicted"]):
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(title)

plt.tight_layout()
out_name = f"patch_gallery_{model_file.replace('.pt','')}.png"
out = Path.home() / f"vlsi-project/{out_name}"
plt.savefig(out, dpi=110)
print("saved", out)
