"""
Round 4 stage B summary: seed spread at c = 128 / capacity 500 (seed 0 from stage A), alpha_evict 6, and the
cluster size x capacity map.

  python -m single_stream.analyse_r4b runs_ss/task1_r4_priority_a_20261025 runs_ss/task1_r4_priority_b_20261025 \
      --fig out.png

Stage B run names: c<c>_k<capacity>_s<seed>__<config>; stage A runs are c 128, capacity 500, seed 0.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from single_stream.analyse_grid import SURFACE, INK, INK2, run_metrics, style

CONFIGS = ["fifo", "fifo_g0.2", "pri_ae1_ar0_g0.2", "pri_ae3_ar0_g0.2", "pri_ae3_ar0_g1", "pri_ae3_ar1_g1"]
LABELS = {"fifo": "FIFO", "fifo_g0.2": "FIFO + gate 0.2", "pri_ae1_ar0_g0.2": "priority ae1 + gate 0.2",
          "pri_ae3_ar0_g0.2": "priority ae3 + gate 0.2", "pri_ae3_ar0_g1": "priority ae3",
          "pri_ae3_ar1_g1": "priority ae3, replay ar1"}
# fixed order: neutral greys for the baselines, categorical hues for the priority configs
COLOURS = {"fifo": "#8a8984", "fifo_g0.2": INK, "pri_ae1_ar0_g0.2": "#2a78d6", "pri_ae3_ar0_g0.2": "#eb6834",
           "pri_ae3_ar0_g1": "#1baf7a", "pri_ae3_ar1_g1": "#8f5bd6"}
KEYS = ["final", "anytime", "worst_digit", "fewshot", "retain", "digits"]


def load(dir_a, dir_b):
    """{(c, capacity, seed, config): metrics}."""
    runs = {}
    for d in glob.glob(os.path.join(dir_a, "*/")):
        name = os.path.basename(d.rstrip("/"))
        if os.path.exists(os.path.join(d, "summary.json")):
            runs[(128, 500, 0, name)] = run_metrics(d)
    for d in glob.glob(os.path.join(dir_b, "*/")):
        m = re.match(r"c(\d+)_k(\d+)_s(\d+)__(.+)$", os.path.basename(d.rstrip("/")))
        if m and os.path.exists(os.path.join(d, "summary.json")):
            runs[(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))] = run_metrics(d)
    return runs


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|---" * len(header) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--fig")
    a = p.parse_args()
    runs = load(a.dir_a, a.dir_b)

    print("Seeds 0-2, c 128, capacity 500: mean (min-max)\n")
    seeded = sorted({k[3] for k in runs if k[:2] == (128, 500) and k[2] > 0})
    rows = []
    for cfg in sorted(seeded, key=lambda c: -np.mean([runs[(128, 500, s, c)]["anytime"] for s in range(3)])):
        ms = [runs[(128, 500, s, cfg)] for s in range(3)]
        rows.append([cfg] + [f"{np.mean([m[k] for m in ms]):.3f} ({min(m[k] for m in ms):.3f}-"
                             f"{max(m[k] for m in ms):.3f})" for k in KEYS])
    table(["config"] + KEYS, rows)

    print("alpha_evict sweep, c 128, capacity 500, seed 0\n")
    rows = []
    for cfg in sorted(k[3] for k in runs if k[:3] == (128, 500, 0) and k[3].startswith("pri_") and "_ar0.5" not in k[3]):
        rows.append([cfg] + [f"{runs[(128, 500, 0, cfg)][k]:.3f}" for k in KEYS])
    table(["config"] + KEYS, rows)

    cells = sorted({k[:2] for k in runs})
    for key in ("anytime", "final", "fewshot", "worst_digit", "digits"):
        print(f"{key} by cluster size / capacity, seed 0\n")
        table(["config"] + [f"c{c} k{k}" for c, k in cells],
              [[cfg] + [f"{runs[(c, k, 0, cfg)][key]:.3f}" if (c, k, 0, cfg) in runs else "-" for c, k in cells]
               for cfg in CONFIGS])

    if a.fig:
        figure(runs, a.fig)


def figure(runs, path):
    sizes = sorted({k[0] for k in runs})
    caps = sorted({k[1] for k in runs})
    metrics_ = [("anytime", "anytime accuracy"), ("final", "final accuracy"), ("fewshot", "few-shot score (n <= 256)")]
    fig, axes = plt.subplots(len(metrics_), len(caps), figsize=(4.6 * len(caps), 3.3 * len(metrics_)), sharey="row",
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, cap in enumerate(caps):
        for row, (key, label) in enumerate(metrics_):
            ax = axes[row][col]
            style(ax)
            for cfg in CONFIGS:
                pts = [(c, runs[(c, cap, 0, cfg)][key]) for c in sizes if (c, cap, 0, cfg) in runs]
                ax.plot(*zip(*pts), marker="o", color=COLOURS[cfg], lw=2, ms=6, label=LABELS[cfg],
                        ls="--" if cfg.startswith("fifo") else "-")
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(c) for c in sizes])
            ax.minorticks_off()
            ax.set_xlabel("cluster size c", color=INK2)
            if col == 0:
                ax.set_ylabel(label, color=INK2)
            if row == 0:
                ax.set_title(f"capacity {cap}", color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Round 4 stage B: replay-only LTM, unpermuted MNIST, 2 epochs, seed 0 (validation)", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
