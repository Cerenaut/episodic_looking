"""
Offline check for wave 9 (Dave, 3 Nov 2026): is exploration noise the bottleneck on problem images? No RL.

For each of the snapshot's 16 training nines, under the what bias:
  reach        the label-supervised optimum p(9) (Adam on the bias), unconstrained, and with the bias limited to
               [-1, 1] (the policy mean is tanh-bounded, so the greedy bias the STM evaluates with cannot leave it)
  odds         from a policy mean m (0, and halfway to the bounded optimum), draws a ~ N(m, sigma^2) for sigma in
               {0.25, 0.5, 1, 2}: the fraction of draws that raise p(9) by more than 0.05 over p(9) at m, and the
               mean best p(9) of 64 draws (one presentation's worth of repeats)

  python -m single_stream.explore_sigma --out runs_ss/stm_frozen_20261101/snap9 --dest runs_ss/stm_frozen_20261101/explore_sigma
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from single_stream.stm_frozen import load_data, load_snapshot

SIGMAS = [0.25, 0.5, 1.0, 2.0]


def p_label(net, x, y, a):
    with torch.no_grad():
        return F.softmax(net(x.expand(len(a), *x.shape[1:]), a), 1)[:, y]


def reach(net, x, y, bounded, steps=300, lr=0.1):
    z = torch.zeros(1, net.bias_size, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        a = torch.tanh(z) if bounded else z
        loss = -F.log_softmax(net(x, a), 1)[0, y]
        opt.zero_grad()
        loss.backward()
        opt.step()
    a = (torch.tanh(z) if bounded else z).detach()
    return a, float(p_label(net, x, y, a)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="snapshot directory")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    ap.add_argument("--preproc", default="runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt")
    ap.add_argument("--new-digit", type=int, default=9)
    ap.add_argument("--new-shots", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--bias-scale", type=float, default=1.3863)
    ap.add_argument("--draws", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.dest, exist_ok=True)
    path = os.path.join(args.dest, "explore_sigma.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; refusing to overwrite")
    xt, yt, _, _ = load_data(args)
    net = load_snapshot(args, "what")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    rows = []
    for j in torch.nonzero(yt == args.new_digit).flatten().tolist():
        x, y = xt[j:j + 1], int(yt[j])
        p0 = float(p_label(net, x, y, torch.zeros(1, net.bias_size))[0])
        _, p_free = reach(net, x, y, bounded=False)
        a_b, p_bounded = reach(net, x, y, bounded=True)
        row = {"index": j, "p0": p0, "reach_free": p_free, "reach_bounded": p_bounded, "odds": {}}
        for start, m in (("zero", torch.zeros(1, net.bias_size)), ("half", 0.5 * a_b)):
            pm = float(p_label(net, x, y, m)[0])
            row["odds"][start] = {"p_mean": pm}
            for s in SIGMAS:
                p = p_label(net, x, y, m + s * torch.randn(args.draws, net.bias_size))
                best64 = p[:(args.draws // 64) * 64].view(-1, 64).max(1).values.mean()
                row["odds"][start][str(s)] = {"improve": float((p > pm + 0.05).float().mean()),
                                              "best_of_64": float(best64)}
        rows.append(row)
        o = row["odds"]
        print(f"{j:6d} p0 {p0:.3f} reach free {p_free:.3f} bounded {p_bounded:.3f} | from 0: improve "
              + " ".join(f"{o['zero'][str(s)]['improve']:.3f}" for s in SIGMAS) + " best64 "
              + " ".join(f"{o['zero'][str(s)]['best_of_64']:.2f}" for s in SIGMAS) + f" | from half (p {o['half']['p_mean']:.2f}): improve "
              + " ".join(f"{o['half'][str(s)]['improve']:.3f}" for s in SIGMAS), flush=True)
    json.dump({"args": vars(args), "sigmas": SIGMAS, "rows": rows}, open(path, "w"))


if __name__ == "__main__":
    main()
