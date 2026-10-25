"""
Round 4 stage B summary: seed spread at c = 128 / capacity 500 (seed 0 from stage A), alpha_evict 6, and the
cluster size x capacity map.

  python -m single_stream.analyse_r4b runs_ss/task1_r4_priority_a_20261025 runs_ss/task1_r4_priority_b_20261025 \
      --fig out.png

Stage B run names: c<c>_k<capacity>_s<seed>__<config>; stage A runs are c 128, capacity 500, seed 0.

--ltm LABEL=DIR ... adds buffer-free LTM baselines (e.g. round 2's LTM-only and continual backprop runs, whose
names end _c<c>_lr..._s<seed>) at the cluster sizes they have, cut to the stage B budget (BUDGET exposures):
final = accuracy at t = BUDGET, anytime = mean over grid checkpoints t <= BUDGET at stage B's spacing (one
epoch / 5). Few-shot (n <= 256) is taken from the full run: those exposures fall in the first epoch.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from single_stream.analyse_grid import EPOCH, SURFACE, INK, INK2, pooled, run_metrics, style

CONFIGS = ["fifo", "fifo_g0.2", "pri_ae1_ar0_g0.2", "pri_ae3_ar0_g0.2", "pri_ae3_ar0_g1", "pri_ae3_ar1_g1"]
LABELS = {"fifo": "FIFO", "fifo_g0.2": "FIFO + gate 0.2", "pri_ae1_ar0_g0.2": "priority ae1 + gate 0.2",
          "pri_ae3_ar0_g0.2": "priority ae3 + gate 0.2", "pri_ae3_ar0_g1": "priority ae3",
          "pri_ae3_ar1_g1": "priority ae3, replay ar1"}
# fixed order: neutral greys for the baselines, categorical hues for the priority configs
COLOURS = {"fifo": "#8a8984", "fifo_g0.2": INK, "pri_ae1_ar0_g0.2": "#2a78d6", "pri_ae3_ar0_g0.2": "#eb6834",
           "pri_ae3_ar0_g1": "#1baf7a", "pri_ae3_ar1_g1": "#8f5bd6"}
KEYS = ["final", "anytime", "worst_digit", "fewshot", "retain", "digits"]
BUDGET = 2 * EPOCH
LTM_STYLE = [("#b8336a", "s"), ("#c98a0b", "D")]  # (colour, marker) per --ltm baseline, in order


def ltm_metrics(d):
    """Metrics of a buffer-free run cut to BUDGET exposures."""
    with open(os.path.join(d, "checkpoints.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    grid = [r for r in recs if r["grid"] and r["t"] <= BUDGET and r["t"] % (EPOCH // 5) == 0]
    end = [r for r in recs if r["t"] == BUDGET][-1]
    return {"final": end["acc"], "anytime": float(np.mean([r["acc"] for r in grid])),
            "worst_digit": min(end["group_acc"]),
            "fewshot": pooled(s["exposure_efficiency_balanced"], lambda lo: lo <= 256)}


def load_ltm(specs):
    """{label: {c: metrics}} from LABEL=DIR specs (DIR's runs named ..._c<c>_lr<lr>_s0)."""
    out = {}
    for spec in specs:
        label, path = spec.split("=", 1)
        out[label] = {}
        for d in glob.glob(path.rstrip("/") + "*/") if not os.path.exists(os.path.join(path, "summary.json")) else [path]:
            m = re.search(r"_c(\d+)_lr[\d.e-]+_s0/?$", d)
            if m:
                out[label][int(m.group(1))] = ltm_metrics(d)
    return out


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
    p.add_argument("--ltm", nargs="*", default=[], help="LABEL=run dir (or path prefix) of buffer-free baselines")
    a = p.parse_args()
    runs = load(a.dir_a, a.dir_b)
    ltm = load_ltm(a.ltm)

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
               for cfg in CONFIGS] +
              [[label] + [f"{v[c][key]:.3f}" if c in v and key in v[c] else "-" for c, k in cells]
               for label, v in ltm.items()])

    if a.fig:
        figure(runs, ltm, a.fig)


def figure(runs, ltm, path):
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
            for (name, v), (colour, marker) in zip(ltm.items(), LTM_STYLE):
                pts = [(c, v[c][key]) for c in sizes if c in v]
                if pts:
                    ax.plot(*zip(*pts), marker=marker, color=colour, lw=2, ms=8, ls=":", label=name,
                            markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(c) for c in sizes])
            ax.minorticks_off()
            ax.set_xlabel("cluster size c", color=INK2)
            if col == 0:
                ax.set_ylabel(label, color=INK2)
            if row == 0:
                ax.set_title(f"capacity {cap}", color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Round 4 stage B: replay-only LTM, unpermuted MNIST, 2 epochs, seed 0 (validation)", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
