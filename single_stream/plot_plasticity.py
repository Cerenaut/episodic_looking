"""
Plasticity stage 1 figure (Dave 16 Nov 2026; plasticity/queue_plast1.sh): end-of-task accuracy on the classes seen
(acc_seen) and the LTM diagnostics (single_stream/plasticity.py) against the task index, one line per model and
cluster size, mean of seeds (rolling mean over 10 tasks for accuracy).

    python -m single_stream.plot_plasticity --root runs_ss/mnist_clustered_single_stream/plasticity/plast1 --out <dir> [--sweep]
(writes <dir>/<root basename>.png)
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = (("ltm", "LTM + sleep", "C0"), ("er", "ER", "C1"), ("ladder1", "LTM alone", "C2"))
PANELS = (("acc_seen", "accuracy on classes seen\n(end of task, rolling 10)"), ("dormant", "dormant FFN units"),
          ("plast_eff_rank", "effective rank of features"), ("plast_weight_norm", "LTM weight norm"))


def ends(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "checkpoints.jsonl")) if '"end": true' in l]
    out = {k: np.array([r[k] for r in rows]) for k in ("acc_seen", "plast_eff_rank", "plast_weight_norm")}
    out["dormant"] = np.array([np.mean([v for k, v in r.items() if k.startswith("plast_dormant")]) for r in rows])
    return out


RATES = ((None, "no continual backprop", "0.2"), ("1e-5", "rate 1e-5", "C2"), ("1e-4", "rate 1e-4", "C1"),
         ("1e-3", "rate 1e-3", "C0"))


def sweep(args):
    """Stage 2: one row per model, the continual-backprop rates against the baseline at c 128, seed 0."""
    fig, axes = plt.subplots(2, len(PANELS), figsize=(16, 7), sharex=True)
    for row, (m, label, _) in enumerate(MODELS[:2]):
        for rate, rlabel, col in RATES:
            run = os.path.join(args.root, f"{m}_c128_s0" if rate is None else f"{m}_cbp{rate}_c128_s0")
            if not os.path.isdir(run):
                continue
            r = ends(run)
            n = len(r["acc_seen"])
            for ax, (key, title) in zip(axes[row], PANELS):
                y, t = r[key], np.arange(1, n + 1)
                if key == "acc_seen":
                    y = np.convolve(y[1:], np.ones(10) / 10, "valid")
                    t = np.arange(11, len(y) + 11)
                ax.plot(t, y, color=col, lw=1.5, label=rlabel)
                ax.set_title(f"{title}, {label}", fontsize=9)
                ax.grid(alpha=.3)
                if key == "plast_weight_norm":
                    ax.set_yscale("log")
        for ax in axes[row]:
            for k in range(100, n + 1, 100):
                ax.axvline(k, color="grey", lw=.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("task")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Plasticity stage 2: continual backprop rate, c 128, seed 0, {n} tasks")
    fig.tight_layout()
    path = os.path.join(args.out, os.path.basename(os.path.normpath(args.root)) + "_sweep.png")
    fig.savefig(path, dpi=120)
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true", help="stage 2: continual-backprop rates at c 128, seed 0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.sweep:
        return sweep(args)
    fig, axes = plt.subplots(2, len(PANELS), figsize=(16, 7), sharex=True)
    for row, c in enumerate((512, 128)):
        for m, label, col in MODELS:
            runs = [ends(r) for r in sorted(glob.glob(os.path.join(args.root, f"{m}_c{c}_s*", "")))]
            if not runs:
                continue
            n = len(runs[0]["acc_seen"])
            for ax, (key, title) in zip(axes[row], PANELS):
                y = np.mean([r[key] for r in runs], 0)
                t = np.arange(1, len(y) + 1)
                if key == "acc_seen":
                    y = np.convolve(y[1:], np.ones(10) / 10, "valid")
                    t = np.arange(11, len(y) + 11)
                ax.plot(t, y, color=col, lw=2, label=label)
                ax.set_title(f"{title}, c {c}", fontsize=9)
                ax.grid(alpha=.3)
        for ax in axes[row]:
            for k in (25, 50, 100, 200, 300, 400, 500):
                if k <= n:
                        ax.axvline(k, color="grey", lw=.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("task")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"Plasticity: {n} tasks (dihedral cycle + deranged label maps), mean of seeds 0-1")
    fig.tight_layout()
    path = os.path.join(args.out, os.path.basename(os.path.normpath(args.root)) + ".png")
    fig.savefig(path, dpi=120)
    print(path)


if __name__ == "__main__":
    main()
