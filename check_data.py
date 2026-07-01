import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DESIGN = Path.home() / "vlsi-project/data/Vortex-small"

def load_npz(path):
    z = np.load(path)
    return z[z.files[0]]

# grab one sample filename from anywhere in the tree
any_npz = next(DESIGN.rglob("*.npz"))
sample_name = any_npz.name
print("sample layout:", sample_name, "\n")

# find every leaf folder that contains this sample
hits = {}
for f in DESIGN.rglob(sample_name):
    feat = f.parent.relative_to(DESIGN)          # full feature path
    arr = load_npz(f)
    hits[str(feat)] = arr
    print(f"{str(feat):60s} shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.3f} max={arr.max():.3f}")

# pick a few to visualize: inputs + the standard GR-horizontal congestion target
want = ["RUDY", "cell_density", "macro_region", "GR_horizontal_overflow"]
show = {k: v for k, v in hits.items()
        if any(w.lower() in k.lower() for w in want)}

n = len(show)
fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
if n == 1: axes = [axes]
for ax, (feat, arr) in zip(axes, show.items()):
    ax.imshow(arr.squeeze(), cmap="viridis")
    ax.set_title(feat.split("/")[-1] + f"\n{arr.squeeze().shape}", fontsize=8)
    ax.axis("off")
plt.tight_layout()
out = Path.home() / "vlsi-project/sample.png"
plt.savefig(out, dpi=110)
print("\nsaved figure ->", out)
