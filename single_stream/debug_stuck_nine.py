"""
Why does a few-shot image never learn under the RL STM? (Dave 2 Nov 2026: training nine 2100 of snap9 stays at
p(9) < 0.16 in every run.) For chosen training images of the frozen snapshot:

  unbiased      the LTM's class probabilities with no bias
  reachable     the label-supervised optimal where bias (Adam on the pre-tanh bias through the frozen LTM, as the
                `bound` command): the most p(y) any bias can give, how far the bias is from 0, how saturated it is
  exploration   p(y) under random biases a ~ N(m, std^2) around m = 0 (the untrained policy) and around fractions of
                the optimal bias: how often exploration finds a better reward, and how large the improvement is
  local slope   the gradient of p(y) w.r.t. the bias at 0, and the p(y) gain along it

  python -m single_stream.debug_stuck_nine --out runs_ss/stm_frozen_20261101/snap9 --images 2100 3765 24602 30329
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from single_stream import datasets
from single_stream.stm_frozen import load_data, load_snapshot


def p_y(net, x, a, y):
    return F.softmax(net(x.expand(len(a), *x.shape[1:]), a), 1)[:, y]


def analyse(net, x, y, std, n_samples, steps, g):
    x = x[None]
    with torch.no_grad():
        p0 = F.softmax(net(x), 1)[0]
    a = torch.zeros(1, net.bias_size, requires_grad=True)
    opt = torch.optim.Adam([a], lr=0.1)
    curve = []
    for s in range(steps):
        loss = -torch.log(p_y(net, x, a, y)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s in (0, 9, 49, 99, steps - 1):
            curve.append(float(torch.exp(-loss.detach())))
    a_opt = a.detach()
    with torch.no_grad():
        p_opt = F.softmax(net(x, a_opt), 1)[0]
    # local slope at a = 0
    z = torch.zeros(1, net.bias_size, requires_grad=True)
    p_y(net, x, z, y).sum().backward()
    grad0 = z.grad[0].clone()
    out = {"label": int(y), "p0": p0.tolist(), "pred0": int(p0.argmax()), "p_y0": float(p0[y]),
           "p_y_opt": float(p_opt[y]), "pred_opt": int(p_opt.argmax()), "opt_curve": curve,
           "opt_norm": float(a_opt.norm()), "opt_abs_mean": float(a_opt.abs().mean()),
           "opt_tanh_sat": float((torch.tanh(a_opt).abs() > 0.9).float().mean()),
           "grad0_norm": float(grad0.norm()), "a_opt": a_opt[0].tolist(), "grad0": grad0.tolist()}
    with torch.no_grad():
        for eps in (0.1, 0.3, 1.0):
            d = grad0 / grad0.norm()
            out[f"p_y_grad_step{eps}"] = float(p_y(net, x, eps * d[None], y))
        explore = {}
        for frac in (0.0, 0.25, 0.5, 1.0):
            m = frac * a_opt
            noise = torch.randn(n_samples, net.bias_size, generator=g) * std
            p = p_y(net, x, m + noise, y)
            pm = float(p_y(net, x, m, y))
            explore[str(frac)] = {"p_mean_bias": pm, "p_sample_mean": float(p.mean()), "p_sample_std": float(p.std()),
                                  "p_sample_max": float(p.max()), "frac_better_0.05": float((p > pm + 0.05).float().mean())}
        out["explore"] = explore
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--images", type=int, nargs="+", required=True)
    ap.add_argument("--data-root", default=os.path.expanduser("~/dev/claude/mnist"))
    ap.add_argument("--preproc", default="runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt")
    ap.add_argument("--new-digit", type=int, default=9)
    ap.add_argument("--new-shots", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--bias-scale", type=float, default=1.3863)
    ap.add_argument("--policy-std", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    os.makedirs(args.dest, exist_ok=True)
    xt, yt, _, _ = load_data(args)
    net = load_snapshot(args, "where")
    shots = torch.nonzero(yt == args.new_digit).flatten().tolist()
    images = args.images if args.images != [-1] else shots
    g = torch.Generator().manual_seed(0)
    res = {}
    for j in images:
        r = analyse(net, xt[j], int(yt[j]), args.policy_std, args.samples, args.steps, g)
        res[j] = r
        e = r["explore"]
        print(f"{j:>6} y {r['label']} pred0 {r['pred0']} p0(y) {r['p_y0']:.3f} | optimal p(y) {r['p_y_opt']:.3f} "
              f"(pred {r['pred_opt']}, |a| {r['opt_abs_mean']:.2f}, sat {r['opt_tanh_sat']:.2f}) | grad0 "
              f"{r['grad0_norm']:.3f}, p(y) after 0.3/1.0 along it {r['p_y_grad_step0.3']:.3f}/{r['p_y_grad_step1.0']:.3f}"
              f" | explore@0: sample max {e['0.0']['p_sample_max']:.3f} better {e['0.0']['frac_better_0.05']:.3f} "
              f"std {e['0.0']['p_sample_std']:.3f}; @0.5 opt: mean {e['0.5']['p_mean_bias']:.3f}")
    json.dump({"args": vars(args), "images": res}, open(os.path.join(args.dest, "reachable.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
