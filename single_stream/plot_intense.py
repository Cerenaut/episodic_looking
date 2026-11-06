"""Intense-sleep probe (Dave, 6 Nov 2026): LTM + replay + sleep with infrequent, deep sleeps and a larger buffer,
against the sweep 1 baselines at the same sleep rate (P 16, buffer 500, seed 0).

    python -m single_stream.plot_intense --run runs_ss/mnist_clustered_single_stream/intense_sleep \
        --base runs_ss/mnist_clustered_single_stream/sweep1 --out <dir>
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_lifelong import task_metrics
from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import save, style

PS = (16, 256, 1024, 4096)
CS = (1, 128, 512)
CAPS = (500, 4096)
RATES = ((1, "-", "1"), (1 / 4, "--", "1/4"))
METRICS = [("final", "Final accuracy", (0.0, 1.0)), ("anytime", "Anytime accuracy", (0.0, 1.0)),
           ("worst", "Worst class (anytime)", (0.0, 1.0)), ("rec_256", "Accuracy 256 after switch", (0.0, 1.0)),
           ("rec_1024", "Accuracy 1,024 after switch", (0.0, 1.0)),
           ("retention_task1", "Retention of task 1", (0.0, 1.0))]
FAM = {"dih": "dih", "lp": "lp"}


def read(d):
    return {"summary": json.load(open(os.path.join(d, "summary.json"))),
            "ckpt": [json.loads(l) for l in open(os.path.join(d, "checkpoints.jsonl"))],
            "fewshot": [json.loads(l) for l in open(os.path.join(d, "fewshot.jsonl"))]}


def load(run_dir, base_dir):
    """{(fam, c, P, K, cap): run}, seed 0; the baselines are sweep 1's P 16 runs (buffer 500)."""
    runs = {}
    for d in glob.glob(os.path.join(run_dir, "*_s0")):
        m = re.match(r"(dih|lp)_c(\d+)_P(\d+)_K(\d+)_B(\d+)_s0$", os.path.basename(d))
        if m and os.path.exists(os.path.join(d, "summary.json")):
            runs[(m.group(1), *map(int, m.groups()[1:]))] = read(d)
    for fam in FAM:
        for c in CS:
            for r, _, _ in RATES:
                d = os.path.join(base_dir, f"{fam}_c{c}_P16_K{int(16 * r)}_s0")
                runs[(fam, c, 16, int(16 * r), 500)] = read(d)
    return runs


def fig_metrics(args, runs):
    fig, axes = plt.subplots(len(METRICS), 6, figsize=(18, 15), facecolor=SURFACE, sharex=True)
    for j, (fam, c) in enumerate([(f, c) for f in FAM for c in CS]):
        for i, (key, title, ylim) in enumerate(METRICS):
            ax = axes[i, j]
            for k, cap in enumerate(CAPS):
                for r, ls, rl in RATES:
                    xs = [P for P in PS if (fam, c, P, int(P * r), cap) in runs]
                    ys = [task_metrics(runs[(fam, c, P, int(P * r), cap)])[key] for P in xs]
                    ax.plot(xs, ys, color=SERIES[k], ls=ls, lw=1.8, marker="o", ms=4,
                            label=f"buffer {cap}, r {rl}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(PS)
            ax.set_xticklabels([str(P) for P in PS])
            ax.set_ylim(*ylim)
            style(ax, f"{fam} c {c}: {title}", xlabel="sleep period P" if i == len(METRICS) - 1 else "")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=10, labelcolor=INK, ncol=len(l))
    fig.suptitle("Intense sleeps: metrics (tasks 2-4) against the sleep period, seed 0. P 16 is sweep 1 "
                 "(buffer 500 only).", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, args, "intense_metrics.png")


def table(args, runs):
    keys = ["zero_shot", "rec_64", "rec_256", "rec_1024", "t90", "anytime", "final", "worst", "retention_task1"]
    lines = ["| fam | c | P | K | buffer | " + " | ".join(keys) + " | buffer age (task 4 end) |",
             "|" + "---|" * (len(keys) + 6)]
    for fam in FAM:
        for c in CS:
            for r, _, _ in RATES:
                for P in PS:
                    for cap in CAPS:
                        run = runs.get((fam, c, P, int(P * r), cap))
                        if run is None:
                            continue
                        m = task_metrics(run)
                        age = run["ckpt"][-1]["buffer_mean_age"]
                        vals = [f"{m[k]:.0f}" if k == "t90" else f"{m[k]:.3f}" for k in keys]
                        lines.append(f"| {fam} | {c} | {P} | {int(P * r)} | {cap} | " + " | ".join(vals)
                                     + f" | {age / 1000:.1f}k |")
    path = os.path.join(args.out, "intense_tables.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.run, args.base)
    print(f"{len(runs)} runs")
    fig_metrics(args, runs)
    table(args, runs)


if __name__ == "__main__":
    main()
