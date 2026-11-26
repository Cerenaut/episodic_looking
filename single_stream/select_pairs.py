"""
CIFAR-100 coarse-pair separability on the frozen preprocessor's features (Dave 26 Nov 2026: choose the pairs in
advance and objectively; one easy, one medium, one hard, plus a development pair for tuning). For every pair of the
20 coarse classes: a binary linear probe (logistic regression, L-BFGS, small L2) on the pair's training-split tokens,
accuracy on its validation images. Training split only: the test set is never touched. Candidates: the pairs nearest
the --quantiles of the accuracy distribution, with no coarse class in two chosen pairs (greedy, in the order given).

    python -m single_stream.select_pairs --preproc runs_ss/preproc/tinyimagenet_conv_c32_s0/preproc.pt \
        --out runs_ss/cifar/pair_selection
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from single_stream import cifar
from single_stream.preprocessor import featurize, load_checkpoint


def probe(x_tr, y_tr, x_va, y_va, l2=1e-3, steps=100):
    """Binary logistic regression on standardised features; validation accuracy."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    a, b = (x_tr - mu) / sd, (x_va - mu) / sd
    w = torch.zeros(a.shape[1], requires_grad=True)
    c = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, c], max_iter=steps, line_search_fn="strong_wolfe")
    yf = y_tr.float()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a @ w + c, yf) + l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return float((((b @ w + c) > 0).long() == y_va).float().mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preproc", required=True)
    p.add_argument("--cifar-root", default=os.path.expanduser("~/dev/claude/cifar-100-python"))
    p.add_argument("--val-per-class", type=int, default=100)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--quantiles", default="0.9,0.5,0.1,0.6", help="easy, medium, hard, development")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    d = cifar.load_cifar100(args.cifar_root)
    x, fine, coarse = d["train"]
    tr, va = cifar.validation_split(fine, args.val_per_class, args.split_seed)
    pre = load_checkpoint(args.preproc)
    feats = featurize(pre, x)
    rows = []
    for a, b in itertools.combinations(range(20), 2):
        mtr = tr[np.isin(coarse[tr], (a, b))]
        mva = va[np.isin(coarse[va], (a, b))]
        acc = probe(feats[mtr], cifar.label(coarse[mtr], (a, b)), feats[mva], cifar.label(coarse[mva], (a, b)))
        rows.append({"pair": [a, b], "names": [d["coarse_names"][a], d["coarse_names"][b]], "val_acc": acc})
        print(f"{a:2d},{b:2d} {acc:.3f}  {d['coarse_names'][a]} vs {d['coarse_names'][b]}", flush=True)
    accs = np.array([r["val_acc"] for r in rows])
    used, chosen = set(), []
    for q, role in zip([float(v) for v in args.quantiles.split(",")], ("easy", "medium", "hard", "development")):
        target = float(np.quantile(accs, q))
        for r in sorted(rows, key=lambda r: abs(r["val_acc"] - target)):
            if not used & set(r["pair"]):
                chosen.append({"role": role, "quantile": q, **r})
                used |= set(r["pair"])
                break
    out = {"args": vars(args), "pairs": sorted(rows, key=lambda r: -r["val_acc"]), "candidates": chosen,
           "summary": {"min": float(accs.min()), "median": float(np.median(accs)), "max": float(accs.max())}}
    json.dump(out, open(os.path.join(args.out, "pairs.json"), "w"), indent=1)
    for c in chosen:
        print(f"{c['role']:12s} q {c['quantile']}: {c['pair']} {c['names'][0]} vs {c['names'][1]}  {c['val_acc']:.3f}")


if __name__ == "__main__":
    main()
