"""
Round 4 stage C: few-shot curves from the triggered evaluations (run_stream fewshot.jsonl), averaged over digits
and seeds, plus final / anytime averaged over seeds.

  python -m single_stream.analyse_fewshot runs_ss/task1_r4_fewshot_c_20261026 --fig curves.png \
      --fig-by-c by_c.png [--seeds 0,1,2]

--fig: few-shot curve E(n) per cluster size (rows) and capacity (columns). --fig-by-c: every metric against
cluster size, one row per metric (anytime, final, few-shot at each threshold n), one column per capacity.

Run names: c<c>_k<capacity>_s<seed>__<config> (buffers) and c<c>_s<seed>__<ltm|ltm_cbp> (no buffer). Each run
gives, per threshold n, the mean over digits of the balanced score when the digit's exposure count first
reached n; the curve is the mean of that over seeds, with the min-max over seeds as a band.
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

from single_stream.analyse_grid import SURFACE, INK, INK2, style

CONFIGS = ["fifo", "fifo_g0.2", "pri_ae3_ar0_g1", "pri_ae3_ar1_g1", "ltm", "ltm_cbp"]
LABELS = {"fifo": "FIFO", "fifo_g0.2": "FIFO + gate 0.2", "pri_ae3_ar0_g1": "priority ae3",
          "pri_ae3_ar1_g1": "priority ae3, replay ar1", "ltm": "LTM only (no buffer)",
          "ltm_cbp": "LTM + continual backprop (eta 0.999)"}
COLOURS = {"fifo": "#8a8984", "fifo_g0.2": INK, "pri_ae3_ar0_g1": "#1baf7a", "pri_ae3_ar1_g1": "#8f5bd6",
           "ltm": "#b8336a", "ltm_cbp": "#c98a0b"}
LINES = {"fifo": "--", "fifo_g0.2": "--", "pri_ae3_ar0_g1": "-", "pri_ae3_ar1_g1": "-", "ltm": ":", "ltm_cbp": ":"}


def load(grid_dir, seeds):
    """{(c, capacity or None, config): {seed: summary}}."""
    runs = {}
    for d in glob.glob(os.path.join(grid_dir, "*/")):
        name = os.path.basename(d.rstrip("/"))
        m = re.match(r"c(\d+)(?:_k(\d+))?_s(\d+)__(.+)$", name)
        if not m or int(m.group(3)) not in seeds or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        with open(os.path.join(d, "summary.json")) as f:
            key = (int(m.group(1)), int(m.group(2)) if m.group(2) else None, m.group(4))
            runs.setdefault(key, {})[int(m.group(3))] = json.load(f)
    return runs


def curve(by_seed, field=1):
    """Thresholds, mean over seeds, min, max of the per-run digit-mean (field 1 = balanced, 2 = recall)."""
    rows = [{r[0]: r[field] for r in s["fewshot_triggered"]} for s in by_seed.values()]
    ths = sorted(set.intersection(*[set(r) for r in rows]))
    vals = np.array([[r[v] for v in ths] for r in rows])
    return np.array(ths), vals.mean(0), vals.min(0), vals.max(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("grid_dir")
    p.add_argument("--fig")
    p.add_argument("--fig-by-c")
    p.add_argument("--seeds", default="0,1,2")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = load(a.grid_dir, seeds)
    sizes = sorted({k[0] for k in runs})
    caps = sorted({k[1] for k in runs if k[1] is not None})

    def get(c, cap, cfg):
        return runs.get((c, None if cfg.startswith("ltm") else cap, cfg))

    show = [1, 8, 16, 32, 64, 128, 256, 512]
    for field, title in ((1, "balanced score"), (2, "recall")):
        print(f"Few-shot {title} at a digit's n-th exposure, mean over digits and seeds (seeds present)\n")
        print("| c | capacity | config | seeds | " + " | ".join(f"n={v}" for v in show) + " |")
        print("|---" * (4 + len(show)) + "|")
        for c in sizes:
            for cap in caps:
                for cfg in CONFIGS:
                    r = get(c, cap, cfg)
                    if not r or (cfg.startswith("ltm") and cap != caps[0]):
                        continue
                    ths, mean, lo, hi = curve(r, field)
                    vals = dict(zip(ths, mean))
                    print(f"| {c} | {'-' if cfg.startswith('ltm') else cap} | {cfg} | {len(r)} | " +
                          " | ".join(f"{vals[v]:.3f}" if v in vals else "-" for v in show) + " |")
        print()

    print("Final / anytime, mean (min-max) over seeds\n")
    print("| c | capacity | config | seeds | final | anytime |")
    print("|---" * 6 + "|")
    for c in sizes:
        for cap in caps:
            for cfg in CONFIGS:
                r = get(c, cap, cfg)
                if not r or (cfg.startswith("ltm") and cap != caps[0]):
                    continue
                f = [s["final_acc"] for s in r.values()]
                t = [s["anytime_acc"] for s in r.values()]
                print(f"| {c} | {'-' if cfg.startswith('ltm') else cap} | {cfg} | {len(r)} | {np.mean(f):.3f} "
                      f"({min(f):.3f}-{max(f):.3f}) | {np.mean(t):.3f} ({min(t):.3f}-{max(t):.3f}) |")

    if a.fig:
        figure(runs, sizes, caps, get, a.fig, seeds)
    if a.fig_by_c:
        figure_by_c(runs, sizes, caps, get, a.fig_by_c, seeds)


def metric_values(by_seed, key):
    """Per-seed values of 'final', 'anytime' or a few-shot threshold (int: balanced score at n = key)."""
    if isinstance(key, str):
        return [s[f"{key}_acc"] for s in by_seed.values()]
    return [r[1] for s in by_seed.values() for r in s["fewshot_triggered"] if r[0] == key]


def figure_by_c(runs, sizes, caps, get, path, seeds):
    ths = sorted({r[0] for v in runs.values() for s in v.values() for r in s["fewshot_triggered"]})
    rows = [("anytime", "anytime accuracy"), ("final", "final accuracy")] + \
           [(v, f"few-shot, n = {v}") for v in ths]
    fig, axes = plt.subplots(len(rows), len(caps), figsize=(4.6 * len(caps), 2.5 * len(rows)), sharey="row",
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, cap in enumerate(caps):
        for row, (key, label) in enumerate(rows):
            ax = axes[row][col]
            style(ax)
            if not isinstance(key, str):
                ax.axhline(0.5, color=INK2, lw=1, alpha=0.5)
            for cfg in CONFIGS:
                pts = []
                for c in sizes:
                    r = get(c, cap, cfg)
                    vals = metric_values(r, key) if r else []
                    if vals:
                        pts.append((c, np.mean(vals), min(vals), max(vals)))
                if not pts:
                    continue
                cs, mean, lo, hi = map(np.array, zip(*pts))
                ax.fill_between(cs, lo, hi, color=COLOURS[cfg], alpha=0.15, lw=0)
                ax.plot(cs, mean, marker="o", ms=5, lw=2, color=COLOURS[cfg], ls=LINES[cfg], label=LABELS[cfg])
            if not isinstance(key, str):  # keep a minimum span so tiny differences near chance don't look large
                lo_, hi_ = ax.get_ylim()
                ax.set_ylim(min(lo_, 0.48), max(hi_, 0.6))
            ax.set_xscale("log", base=2)
            ax.set_xticks(sizes, [str(c) for c in sizes])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(f"capacity {cap}", color=INK)
            if row == len(rows) - 1:
                ax.set_xlabel("cluster size c", color=INK2)
            if col == 0:
                ax.set_ylabel(label, color=INK2)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Round 4 stage C: all models vs cluster size; few-shot = balanced score when a digit reaches its "
                 f"n-th exposure (mean over digits; chance 0.5)\nline = mean over seeds "
                 f"{','.join(map(str, seeds))}, band = min-max", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    fig.savefig(path, dpi=110, facecolor=SURFACE)


def figure(runs, sizes, caps, get, path, seeds):
    fig, axes = plt.subplots(len(sizes), len(caps), figsize=(4.6 * len(caps), 3.4 * len(sizes)), sharey=True,
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, c in enumerate(sizes):
        for col, cap in enumerate(caps):
            ax = axes[row][col]
            style(ax)
            ax.axhline(0.5, color=INK2, lw=1, ls="-", alpha=0.5)
            for cfg in CONFIGS:
                r = get(c, cap, cfg)
                if not r:
                    continue
                ths, mean, lo, hi = curve(r)
                ax.fill_between(ths, lo, hi, color=COLOURS[cfg], alpha=0.15, lw=0)
                ax.plot(ths, mean, marker="o", ms=5, lw=2, color=COLOURS[cfg], ls=LINES[cfg], label=LABELS[cfg])
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
            ax.minorticks_off()
            if row == 0:
                ax.set_title(f"capacity {cap}", color=INK)
            if row == len(sizes) - 1:
                ax.set_xlabel("exposures of the digit so far (n)", color=INK2)
            if col == 0:
                ax.set_ylabel(f"c = {c}\nbalanced score", color=INK2)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Few-shot: balanced score when a digit reaches its n-th exposure (mean over digits; line = mean "
                 f"over seeds {','.join(map(str, seeds))}, band = min-max; chance 0.5)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)


if __name__ == "__main__":
    main()
