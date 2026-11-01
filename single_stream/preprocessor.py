"""
Frozen convolutional preprocessor: turns a 28x28 image into a 7x7 grid of feature tokens for the transformer LTM.

Pre-trained once, iid, on a different dataset with similar content (Omniglot background set for MNIST; Dave,
1 Nov 2026), then frozen permanently: every model and experiment uses the same checkpoint. Its weights are learned
from data, so they count as values in the memory metric (plan.md section 4), reported separately.

Pre-training (classification of the 964 background characters; the linear head is discarded afterwards):
  python -m single_stream.preprocessor --out runs_ss/preproc/omniglot_conv_s0

Omniglot images (105x105, black strokes on white) are inverted and resized to 28x28 so they look like MNIST
(bright strokes on black, values in [0, 1]).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConvPreprocessor(nn.Module):
    """(N, 1, 28, 28) -> (N, 49, channels) tokens: three 3x3 convs, two with stride 2, each followed by GroupNorm
    and ReLU, then a 1x1 conv to `channels` (ReLU). GroupNorm normalises each image on its own, so the frozen map is
    still a fixed function of the single image (no batch statistics)."""

    def __init__(self, channels=32, width=64):
        super().__init__()
        self.channels = channels
        self.width = width
        self.body = nn.Sequential(
            nn.Conv2d(1, width // 2, 3, padding=1), nn.GroupNorm(8, width // 2), nn.ReLU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.ReLU(),   # 14x14
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.ReLU(),        # 7x7
            nn.Conv2d(width, channels, 1), nn.ReLU())

    def forward(self, x):
        x = x.reshape(len(x), 1, 28, 28)
        f = self.body(x)                                # (N, C, 7, 7)
        return f.flatten(2).transpose(1, 2)             # (N, 49, C)


class PretrainNet(nn.Module):
    """Pre-training head on the flattened tokens (keeps position-specific features, as the transformer has position
    embeddings); discarded after pre-training."""

    def __init__(self, pre, num_classes):
        super().__init__()
        self.pre = pre
        self.head = nn.Linear(49 * pre.channels, num_classes)

    def forward(self, x):
        return self.head(self.pre(x).flatten(1))


def load_omniglot(root, background=True):
    """Omniglot as (N, 784) float in [0, 1], MNIST-like (inverted, 28x28), and character labels (N,)."""
    from torchvision.datasets import Omniglot

    d = Omniglot(root, background=background, download=False)
    xs, ys = [], []
    for img, y in d:
        a = 1.0 - torch.from_numpy(np.asarray(img, dtype=np.float32))     # strokes bright
        xs.append(a)
        ys.append(y)
    x = torch.stack(xs)[:, None]                                          # (N, 1, 105, 105)
    x = F.interpolate(x, size=(28, 28), mode="area").clamp(0, 1)
    return x.reshape(len(x), -1), torch.tensor(ys)


def augment(x, gen, max_shift=2, max_rot=10.0, max_scale=0.1):
    """Random small affine transforms of (N, 784) images."""
    n = len(x)
    img = x.reshape(n, 1, 28, 28)
    ang = (torch.rand(n, generator=gen) * 2 - 1) * max_rot * np.pi / 180
    sc = 1 + (torch.rand(n, generator=gen) * 2 - 1) * max_scale
    sh = (torch.rand(n, 2, generator=gen) * 2 - 1) * max_shift * 2 / 28
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = torch.cos(ang) / sc
    theta[:, 0, 1] = -torch.sin(ang) / sc
    theta[:, 1, 0] = torch.sin(ang) / sc
    theta[:, 1, 1] = torch.cos(ang) / sc
    theta[:, :, 2] = sh
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, align_corners=False).reshape(n, -1)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    pre = ConvPreprocessor(**ck["config"])
    pre.load_state_dict(ck["state_dict"])
    pre.eval()
    for p in pre.parameters():
        p.requires_grad_(False)
    return pre


@torch.no_grad()
def featurize(pre, x, batch=2000):
    """Frozen tokens for (N, 784) images: (N, 49 * C) float32, flattened row-major (token, channel)."""
    out = [pre(x[i:i + batch]).reshape(min(batch, len(x) - i), -1) for i in range(0, len(x), batch)]
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=os.path.expanduser("~/dev/claude/omniglot"))
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-per-class", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if os.path.exists(os.path.join(args.out, "preproc.pt")):
        raise SystemExit(f"{args.out} already has a checkpoint; refusing to overwrite")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    x, y = load_omniglot(args.root, background=True)
    num_classes = int(y.max()) + 1
    rng = np.random.default_rng(args.seed)
    val = np.concatenate([rng.choice(np.flatnonzero(y.numpy() == c), args.val_per_class, replace=False)
                          for c in range(num_classes)])
    tr = np.setdiff1d(np.arange(len(y)), val)
    xtr, ytr, xva, yva = x[tr], y[tr], x[val], y[val]

    pre = ConvPreprocessor(args.channels, args.width)
    net = PretrainNet(pre, num_classes)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    log = open(os.path.join(args.out, "log.txt"), "a")
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(ytr), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss = F.cross_entropy(net(augment(xtr[idx], gen)), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(xva).argmax(1) == yva).float().mean())
        history.append({"epoch": epoch, "train_loss": tot / len(ytr), "val_acc": acc})
        msg = f"epoch {epoch}  loss {tot / len(ytr):.3f}  val acc {acc:.4f}  {time.time() - t0:.0f}s"
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    torch.save({"config": {"channels": args.channels, "width": args.width}, "state_dict": pre.state_dict()},
               os.path.join(args.out, "preproc.pt"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"history": history, "values": sum(p.numel() for p in pre.parameters())}, f, indent=1)


if __name__ == "__main__":
    main()
