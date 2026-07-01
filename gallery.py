import torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from train import CircuitNet, ResUNetSE, DEVICE

ds = CircuitNet()

net = ResUNetSE().to(DEVICE)
net.load_state_dict(torch.load(Path.home() / "vlsi-project/unet.pt",
                               map_location=DEVICE, weights_only=True))
net.eval()

N = 5                                   # how many different chips to show
fig, axes = plt.subplots(N, 3, figsize=(9, 3 * N))

for row in range(N):
    i = row * 700                       # spread picks across the dataset
    x, y = ds[i]
    with torch.no_grad():
        pred = net(x[None].to(DEVICE)).cpu()[0, 0]
    axes[row, 0].imshow(x[1], cmap="viridis")
    axes[row, 1].imshow(y[0], cmap="viridis")
    axes[row, 2].imshow(pred, cmap="viridis")
    for col, title in zip(range(3), ["input RUDY", "true congestion", "predicted"]):
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(title)

plt.tight_layout()
out = Path.home() / "vlsi-project/gallery.png"
plt.savefig(out, dpi=110)
print("saved", out)
