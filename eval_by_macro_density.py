"""
NOVELTY EXPERIMENT — Phase 1
Does congestion-prediction accuracy depend on macro density?

Splits the validation set into buckets by how much macro area each patch
contains, then reports Pearson separately per bucket. No retraining needed —
just runs the existing best model over validation data.

Run:  python eval_by_macro_density.py
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from train_patch128_v2 import PatchCircuitNet, ResUNetSE, DEVICE, IN_FEATS

MODEL_FILE = "unet_patch128_v2.pt"

# macro-density buckets: (label, lower bound, upper bound) as fraction of patch
BUCKETS = [
    ("0% (no macro)",   -0.001, 0.001),
    ("0-10% (sparse)",   0.001, 0.10),
    ("10-30% (medium)",  0.10,  0.30),
    ("30%+ (dense)",     0.30,  1.01),
]

def pearson(p, t):
    p, t = p.flatten(), t.flatten()
    p, t = p - p.mean(), t - t.mean()
    denom = p.norm() * t.norm()
    if denom < 1e-8:            # flat patch -> correlation undefined, skip
        return None
    return (p @ t / denom).item()

def nrms(p, t):
    rng = t.max() - t.min()
    if rng < 1e-4:
        return None
    return (torch.sqrt(((p - t) ** 2).mean()) / rng).item()

def main():
    ds = PatchCircuitNet()
    n_val = max(1, int(0.15 * len(ds)))
    _, va = random_split(ds, [len(ds) - n_val, n_val],
                         generator=torch.Generator().manual_seed(0))  # SAME split as training
    vl = DataLoader(va, batch_size=16)

    net = ResUNetSE(cin=len(IN_FEATS)).to(DEVICE)
    net.load_state_dict(torch.load(Path.home() / f"vlsi-project/{MODEL_FILE}",
                                   map_location=DEVICE, weights_only=True))
    net.eval()

    # collect per-patch: macro density, pearson, nrms
    records = []
    with torch.no_grad():
        for x, y in vl:
            out = net(x.to(DEVICE)).cpu()
            for i in range(out.size(0)):
                macro_channel = x[i, 0]                       # channel 0 = macro_region
                density = (macro_channel > 0).float().mean().item()
                p = pearson(out[i, 0], y[i, 0])
                n = nrms(out[i, 0], y[i, 0])
                records.append((density, p, n))

    print(f"\nEvaluated {len(records)} validation patches")
    print(f"Model: {MODEL_FILE}\n")
    print(f"{'bucket':<20} {'patches':>8} {'Pearson':>9} {'NRMS':>9}")
    print("-" * 50)

    overall_p, overall_n = [], []

    for label, lo, hi in BUCKETS:
        in_bucket = [(d, p, n) for (d, p, n) in records if lo <= d < hi]
        if not in_bucket:
            print(f"{label:<20} {0:>8}        --        --")
            continue
        ps = [p for (_, p, _) in in_bucket if p is not None]
        ns = [n for (_, _, n) in in_bucket if n is not None]
        overall_p += ps
        overall_n += ns
        mean_p = sum(ps) / len(ps) if ps else float("nan")
        mean_n = sum(ns) / len(ns) if ns else float("nan")
        print(f"{label:<20} {len(in_bucket):>8} {mean_p:>9.4f} {mean_n:>9.4f}")

    print("-" * 50)
    if overall_p:
        print(f"{'OVERALL':<20} {len(records):>8} "
              f"{sum(overall_p)/len(overall_p):>9.4f} "
              f"{sum(overall_n)/len(overall_n):>9.4f}")

    # how much of the validation set is macro-free? (context for the finding)
    zero_frac = sum(1 for (d, _, _) in records if d < 0.001) / len(records)
    print(f"\nMacro-free patches make up {100*zero_frac:.1f}% of the validation set.")

if __name__ == "__main__":
    main()
