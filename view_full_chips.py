"""
View FULL, WHOLE chips (native resolution, no patch-cropping) to see the
real macro layout directly -- not a small 128x128 window.
Shows macro_region, RUDY, and true congestion for several different samples,
and prints each one's exact pixel dimensions (proving they're NOT all the
same fixed size).
"""
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = Path.home() / "vlsi-project/data"
DESIGNS = ["Vortex-small", "nvdla-small", "zero-riscy"]

TGT_H = "congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow"
TGT_V = "congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow"

def load(path):
    z = np.load(path)
    return z[z.files[0]].astype(np.float32)

# Collect a handful of full samples, one or two per design, to compare
samples = []
for d in DESIGNS:
    base = DATA / d
    tgt_dir = base / TGT_H
    if not tgt_dir.exists():
        continue
    files = sorted(tgt_dir.glob("*.npz"))
    # take 2 samples spread apart from this design
    picks = [files[0], files[len(files) // 2]] if len(files) > 1 else files
    for f in picks:
        name = f.name
        macro_path = base / "macro_region" / name
        rudy_path = base / "RUDY" / "RUDY" / name
        h_path = base / TGT_H / name
        v_path = base / TGT_V / name
        if all(p.exists() for p in [macro_path, rudy_path, h_path, v_path]):
            samples.append((d, name, macro_path, rudy_path, h_path, v_path))

print(f"Showing {len(samples)} full whole-chip samples\n")

# HIGH QUALITY combined comparison: no smoothing, high dpi
fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))
if len(samples) == 1:
    axes = axes[None, :]

for row, (design, name, macro_path, rudy_path, h_path, v_path) in enumerate(samples):
    macro = load(macro_path)
    rudy = load(rudy_path)
    congestion = load(h_path) + load(v_path)

    print(f"[{design}] {name}")
    print(f"    shape: {macro.shape}  (this chip's real pixel dimensions)")
    print(f"    macro nonzero pixels: {(macro > 0).sum()} / {macro.size} "
          f"({100*(macro > 0).sum()/macro.size:.1f}% of chip)")

    # interpolation="nearest" = show REAL pixel values, no blur/smoothing
    axes[row, 0].imshow(macro, cmap="viridis", interpolation="nearest")
    axes[row, 1].imshow(rudy, cmap="viridis", interpolation="nearest")
    axes[row, 2].imshow(congestion, cmap="viridis", interpolation="nearest")
    for col in range(3):
        axes[row, col].axis("off")
    if row == 0:
        axes[row, 0].set_title("macro_region (FULL chip)", fontsize=14)
        axes[row, 1].set_title("RUDY (FULL chip)", fontsize=14)
        axes[row, 2].set_title("true congestion (FULL chip)", fontsize=14)
    axes[row, 0].text(5, 20, f"{design}\n{macro.shape}", color="white",
                       fontsize=11, va="top", weight="bold")

plt.tight_layout()
out = Path.home() / "vlsi-project/full_chip_comparison.png"
plt.savefig(out, dpi=200)   # much higher dpi than before
print(f"\nsaved {out}  (high-dpi combined view)")

# ALSO save each chip's macro_region individually at TRUE native resolution
# (one pixel in the file = one pixel in the image, zero scaling/blur at all)
out_dir = Path.home() / "vlsi-project/full_res_singles"
out_dir.mkdir(exist_ok=True)
for design, name, macro_path, rudy_path, h_path, v_path in samples:
    macro = load(macro_path)
    h, w = macro.shape
    dpi = 100
    fig2 = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax2 = fig2.add_axes([0, 0, 1, 1])   # no margins, no axis, pure pixels
    ax2.imshow(macro, cmap="viridis", interpolation="nearest")
    ax2.axis("off")
    safe_name = name.replace(".npz", "")
    single_out = out_dir / f"{design}_{safe_name}_macro_TRUERES.png"
    plt.savefig(single_out, dpi=dpi)
    plt.close(fig2)
    print(f"    saved true-resolution single image: {single_out}")

print("\nDone. For the sharpest possible view, open the files in "
      "full_res_singles/ -- those are pixel-for-pixel exact, no scaling at all.")
