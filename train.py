"""
Congestion predictor for CircuitNet-N14.
Residual U-Net + SE blocks, GroupNorm (stable at small batch), weighted-MSE loss.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATA = Path.home() / "vlsi-project/data"
DESIGNS = ["Vortex-small", "nvdla-small", "zero-riscy"]

SIZE = 384
BATCH_SIZE = 4
NUM_WORKERS = 0
EPOCHS = 6
LR = 2e-4
WEIGHT = 20.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_FEATS = ["macro_region", "RUDY/RUDY", "RUDY/RUDY_pin"]
TGT_H = "congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow"
TGT_V = "congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow"

def load(path):
    z = np.load(path); return z[z.files[0]].astype(np.float32)

def resize(a):
    t = torch.from_numpy(a)[None, None]
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return t[0, 0]

def norm(t):
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo + 1e-8)

class CircuitNet(Dataset):
    def __init__(self):
        self.items = []
        for d in DESIGNS:
            base = DATA / d
            tgt_dir = base / TGT_H
            if not tgt_dir.exists():
                continue
            for f in sorted(tgt_dir.glob("*.npz")):
                name = f.name
                paths = {k: base / k / name for k in IN_FEATS}
                paths["h"] = base / TGT_H / name
                paths["v"] = base / TGT_V / name
                if all(p.exists() for p in paths.values()):
                    self.items.append(paths)
        print(f"dataset size: {len(self.items)} samples")
        print("pre-loading all data into RAM (one-time cost)...")
        self.cache = []
        for p in self.items:
            chans = [norm(resize(load(p[k]))) for k in IN_FEATS]
            x = torch.stack(chans)
            y = resize(load(p["h"])) + resize(load(p["v"]))
            self.cache.append((x, y[None]))
        print("done caching.")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.cache[i]

# GroupNorm: normalizes within feature groups, NOT across the batch.
# This is the fix -- it's stable at batch size 8, unlike BatchNorm which
# collapsed on small batches of sparse high-res maps.
def gn(ch):
    groups = 8 if ch >= 8 else 1
    return nn.GroupNorm(groups, ch)

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 4), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 4), ch, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResSEBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = gn(cout)
        self.se = SE(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x):
        idn = self.skip(x)
        h = F.relu(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + idn)

class ResUNetSE(nn.Module):
    def __init__(self, cin=3, base=32):
        super().__init__()
        self.e1 = ResSEBlock(cin, base)
        self.e2 = ResSEBlock(base, base*2)
        self.e3 = ResSEBlock(base*2, base*4)
        self.bott = ResSEBlock(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2); self.d3 = ResSEBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2); self.d2 = ResSEBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2);   self.d1 = ResSEBlock(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d = self.d3(torch.cat([self.up3(b), e3], 1))
        d = self.d2(torch.cat([self.up2(d), e2], 1))
        d = self.d1(torch.cat([self.up1(d), e1], 1))
        return self.out(d)

def loss_fn(pred, target):
    w = 1.0 + WEIGHT * target
    return (w * (pred - target) ** 2).mean()

def main():
    print("device:", DEVICE, "| SIZE:", SIZE, "| batch:", BATCH_SIZE,
          "| epochs:", EPOCHS, "| LR:", LR, "| WEIGHT:", WEIGHT)
    ds = CircuitNet()
    n_val = max(1, int(0.15 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=NUM_WORKERS, pin_memory=True)
    vl = DataLoader(va, batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS, pin_memory=True)

    net = ResUNetSE().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val = float("inf")
    for ep in range(EPOCHS):
        net.train(); tot = 0
        for x, y in tl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                out = net(x); loss = loss_fn(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * x.size(0)
        sched.step()

        net.eval(); vtot = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    vtot += loss_fn(net(x), y).item() * x.size(0)
        vavg = vtot / len(va)
        flag = ""
        if vavg < best_val:
            best_val = vavg
            torch.save(net.state_dict(), Path.home() / "vlsi-project/unet.pt")
            flag = "  <- saved best"
        print(f"epoch {ep+1:3d}  train {tot/len(tr):.5f}  val {vavg:.5f}{flag}")

    net.load_state_dict(torch.load(Path.home() / "vlsi-project/unet.pt",
                                   map_location=DEVICE, weights_only=True))
    net.eval()
    x, y = va[0]
    with torch.no_grad():
        pred = net(x[None].to(DEVICE)).cpu()[0, 0]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, img, t in zip(ax, [x[1], y[0], pred], ["input RUDY", "true congestion", "predicted"]):
        a.imshow(img, cmap="viridis"); a.set_title(t); a.axis("off")
    plt.tight_layout(); out = Path.home() / "vlsi-project/prediction.png"
    plt.savefig(out, dpi=110); print("saved", out)
    print("best val loss:", round(best_val, 5))

if __name__ == "__main__":
    main()
