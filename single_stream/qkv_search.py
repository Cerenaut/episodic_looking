"""
Offline search over bias families on the frozen gap snapshot, no RL (Dave, 2 Nov 2026, wave 8 item 3): which
interventions in the attention block flip an image to its label with the fewest bias elements?

Families (each element a bounded log-gain s tanh(a), s = bias_scale; model/transformer.py): where (49 key tokens),
what (64, the shared gain before the Q/K/V projections), and per-channel gains on the projected queries, keys and
values: q, k, v (64 each), qk, qv, kv (128), qkv (192).

For each image and family:
  dense       optimise the whole bias for log p(y) (Adam), the label-supervised optimum
  top-k       keep the k elements with the largest |s tanh(a)|, re-optimise only those (k = 1, 2, 4, ..., all)
  L1 sweep    optimise log p(y) - lambda sum |s tanh(a)| (cross-check that top-k is not misleading): elements
              with |s tanh(a)| > 0.1, and success
Reported: success (argmax = y) and p(y) vs k, the median k to flip, success at k <= 8, the total |log-gain|.
Images: the snapshot's 16 training nines, and held-out validation nines the unbiased LTM gets wrong.

  python -m single_stream.qkv_search --out runs_ss/stm_frozen_20261101/snap9 --dest runs_ss/stm_frozen_20261101/qkv_search
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from single_stream.stm_frozen import load_data, load_snapshot

FAMILIES = ["where", "what", "q", "k", "v", "qk", "qv", "kv", "qkv"]
KS = [1, 2, 4, 8, 16, 32, 64]


def optimise(net, x, y, a0, mask, steps, lr, l1=0.0):
    """Adam on the bias for log p(y) - l1 * sum |s tanh(a)|, elements outside `mask` held at 0. Batched."""
    a = (a0 * mask).clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        logp = F.log_softmax(net(x, a * mask), 1).gather(1, y[:, None])[:, 0]
        loss = -logp.sum()
        if l1 > 0:
            loss = loss + l1 * (net.bias_scale * torch.tanh(a * mask)).abs().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = a.detach() * mask
        p = F.softmax(net(x, a), 1)
    return a, p


def analyse(net, x, y, args):
    out = {}
    n, size = len(y), net.bias_size
    ones = torch.ones(n, size)
    a_dense, p = optimise(net, x, y, torch.zeros(n, size), ones, args.steps, args.lr)
    g = (net.bias_scale * torch.tanh(a_dense)).abs()
    out["dense"] = {"p": p.gather(1, y[:, None])[:, 0].tolist(), "success": (p.argmax(1) == y).tolist(),
                    "abs_gain_sum": g.sum(1).tolist()}
    order = g.argsort(dim=1, descending=True)
    out["topk"] = {}
    for k in [k for k in KS if k < size] + [size]:
        mask = torch.zeros(n, size).scatter_(1, order[:, :k], 1.0)
        a, p = optimise(net, x, y, a_dense, mask, args.refine_steps, args.lr)
        out["topk"][str(k)] = {"p": p.gather(1, y[:, None])[:, 0].tolist(), "success": (p.argmax(1) == y).tolist(),
                               "abs_gain_sum": (net.bias_scale * torch.tanh(a)).abs().sum(1).tolist()}
    out["l1"] = {}
    for lam in args.l1:
        a, p = optimise(net, x, y, torch.zeros(n, size), ones, args.steps, args.lr, l1=lam)
        gl = (net.bias_scale * torch.tanh(a)).abs()
        out["l1"][str(lam)] = {"success": (p.argmax(1) == y).tolist(), "active": (gl > 0.1).sum(1).tolist(),
                               "p": p.gather(1, y[:, None])[:, 0].tolist()}
    return out


def k_to_flip(res):
    """Per image: the smallest k at which the top-k bias flips it (None if even the dense one does not)."""
    ks = sorted(res["topk"], key=int)
    n = len(res["dense"]["p"])
    return [next((int(k) for k in ks if res["topk"][k]["success"][i]), None) for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="snapshot directory")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    ap.add_argument("--preproc", default="runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt")
    ap.add_argument("--new-digit", type=int, default=9)
    ap.add_argument("--new-shots", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--bias-scale", type=float, default=1.3863)
    ap.add_argument("--held-out", type=int, default=100, help="validation nines misclassified unbiased")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--refine-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l1", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.dest, exist_ok=True)
    path = os.path.join(args.dest, "qkv_search.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    xt, yt, xv, yv = load_data(args)
    shots = torch.nonzero(yt == args.new_digit).flatten()
    with torch.no_grad():
        base = load_snapshot(args, "where")
        val9 = torch.nonzero(yv == args.new_digit).flatten()
        wrong = val9[base(xv[val9]).argmax(1) != args.new_digit][:args.held_out]
    sets = {"train": (xt[shots], yt[shots], shots.tolist()), "held_out": (xv[wrong], yv[wrong], wrong.tolist())}
    result = {"args": vars(args), "sets": {k: v[2] for k, v in sets.items()}, "families": {}}
    for fam in args.families:
        net = load_snapshot(args, fam)
        result["families"][fam] = {"size": net.bias_size}
        for name, (x, y, _) in sets.items():
            t0 = time.time()
            res = analyse(net, x, y, args)
            res["k_to_flip"] = k_to_flip(res)
            result["families"][fam][name] = res
            kf = [k for k in res["k_to_flip"] if k is not None]
            print(f"{fam:5s} {name:8s} n {len(y):3d}  dense success {np.mean(res['dense']['success']):.2f}  "
                  f"flipped {len(kf)}/{len(y)}  median k {np.median(kf) if kf else float('nan'):.0f}  "
                  f"success k<=8 {np.mean(res['topk']['8']['success']) if '8' in res['topk'] else float('nan'):.2f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    json.dump(result, open(path, "w"))


if __name__ == "__main__":
    main()
