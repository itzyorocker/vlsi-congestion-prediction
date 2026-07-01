"""
Evaluate the patch-trained model on the validation patches.
Reports NRMS, SSIM, Pearson -- same metrics as eval.py, but on the
native-resolution patch dataset and loading unet_patch.pt.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from train_patch import PatchCircuitNet, ResUNetSE, DEVICE

def nrms(p, t):
    rmse = torch.sqrt(((p - t) ** 2).mean())
    return (rmse / (t.max() - t.min() + 1e-8)).item()

def pearson(p, t):
    p, t = p.flatten(), t.flatten()
    p, t = p - p.mean(), t - t.mean()
    return (p @ t / (p.norm() * t.norm() + 1e-8)).item()

def _gauss_win(ks=11, sigma=1.5):
    coords = torch.arange(ks, dtype=torch.float32) - ks // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2)); g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]

def ssim(p, t, win):
    p = p[None, None]; t = t[None, None]
    C1, C2 = 0.01**2, 0.03**2
    mu_p = F.conv2d(p, win, padding=5); mu_t = F.conv2d(t, win, padding=5)
    mu_p2, mu_t2, mu_pt = mu_p*mu_p, mu_t*mu_t, mu_p*mu_t
    sig_p = F.conv2d(p*p, win, padding=5) - mu_p2
    sig_t = F.conv2d(t*t, win, padding=5) - mu_t2
    sig_pt = F.conv2d(p*t, win, padding=5) - mu_pt
    s = ((2*mu_pt + C1)*(2*sig_pt + C2)) / ((mu_p2 + mu_t2 + C1)*(sig_p + sig_t + C2))
    return s.mean().item()

def main():
    ds = PatchCircuitNet()
    n_val = max(1, int(0.15 * len(ds)))
    _, va = random_split(ds, [len(ds) - n_val, n_val],
                         generator=torch.Generator().manual_seed(0))  # SAME split as training
    vl = DataLoader(va, batch_size=8)

    net = ResUNetSE().to(DEVICE)
    net.load_state_dict(torch.load(Path.home() / "vlsi-project/unet_patch.pt",
                                   map_location=DEVICE, weights_only=True))
    net.eval()

    win = _gauss_win()
    tot_n, tot_p, tot_s, count = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in vl:
            out = net(x.to(DEVICE)).cpu()
            for i in range(out.size(0)):
                tot_n += nrms(out[i, 0], y[i, 0])
                tot_p += pearson(out[i, 0], y[i, 0])
                tot_s += ssim(out[i, 0], y[i, 0], win)
                count += 1

    print(f"\nPATCH MODEL  ->  NRMS {tot_n/count:.4f}   "
          f"SSIM {tot_s/count:.4f}   Pearson {tot_p/count:.4f}")

if __name__ == "__main__":
    main()
