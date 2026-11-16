"""
Plasticity stage 1 figure (Dave 16 Nov 2026; plasticity/queue_plast1.sh): end-of-task accuracy on the classes seen
(acc_seen) and the LTM diagnostics (single_stream/plasticity.py) against the task index, one line per model and
cluster size, mean of seeds (rolling mean over 10 tasks for accuracy).

    python -m single_stream.plot_plasticity --root runs_ss/mnist_clustered_single_stream/plasticity/plast1 --out <dir>
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(2, len(PANELS), figsize=(16, 7), sharex=True)
    for row, c in enumerate((512, 128)):
        for m, label, col in MODELS:
            runs = [ends(r) for r in sorted(glob.glob(os.path.join(args.root, f"{m}_c{c}_s*", "")))]
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
            for k in (25, 50, 100):
                ax.axvline(k, color="grey", lw=.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("task")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Plasticity stage 1: 100 tasks (dihedral cycle + deranged label maps), mean of seeds 0-1")
    fig.tight_layout()
    path = os.path.join(args.out, "plast1.png")
    fig.savefig(path, dpi=120)
    print(path)


if __name__ == "__main__":
    main()
