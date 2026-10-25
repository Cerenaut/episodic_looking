"""
Round 4 (priority buffer search) summary: a table ranked by anytime accuracy and a line plot.

  python -m single_stream.analyse_grid runs_ss/task1_r4_priority_a_20261025 [--fig out.png] [--prefix c32_k250_s0__]

Run names: fifo, fifo_g<floor>, pri_ae<alpha_evict>_ar<alpha_replay>_g<floor>. Columns:
  final, anytime (mean over grid checkpoints), worst grid checkpoint after epoch 1, worst digit at the end;
  fewshot: balanced per-digit score pooled over exposure bins n <= 256 (weighted by count; chance 0.5);
  retain: balanced score pooled over bins of >= 512 exposures since the digit was last seen;
  written: fraction of offered samples stored; digits: mean number of digits present in the buffer over grid
  checkpoints; age: mean age of stored samples (exposures) over grid checkpoints, priority buffer only.
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

EPOCH = 54000
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def pooled(curve, keep):
    rows = [(s, n) for lo, s, n in curve if keep(lo) and s is not None]
    return sum(s * n for s, n in rows) / sum(n for _, n in rows) if rows else np.nan


def run_metrics(d):
    with open(os.path.join(d, "checkpoints.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    with open(os.path.join(d, "summary.json")) as f:
        s = json.load(f)
    grid = [r for r in recs if r["grid"]]
    late = [r["acc"] for r in grid if r["t"] > EPOCH]
    out = {"final": recs[-1]["acc"], "anytime": s["anytime_acc"], "worst": min(late) if late else np.nan,
           "worst_digit": min(recs[-1]["group_acc"]),
           "fewshot": pooled(s["exposure_efficiency_balanced"], lambda lo: lo <= 256),
           "retain": pooled(s["retention_balanced"], lambda lo: lo >= 512),
           "written": s["buffer_written"] / s["buffer_offered"],
           "digits": float(np.mean([sum(1 for c in r["buffer_labels"] if c > 0) for r in grid])),
           "age": float(np.mean([r["buffer_mean_age"] for r in grid])) if "buffer_mean_age" in grid[0] else np.nan}
    return out


def parse(name):
    m = re.match(r"pri_ae([\d.]+)_ar([\d.]+)_g([\d.]+)$", name)
    if m:
        return "priority", float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.match(r"fifo(?:_g([\d.]+))?$", name)
    if m:
        return "fifo", None, None, float(m.group(1) or 1)
    return None


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(rows, path, title):
    floors = sorted({r["floor"] for r in rows}, reverse=True)
    evicts = sorted({r["ae"] for r in rows if r["kind"] == "priority"})
    colours = dict(zip(evicts, [BLUE, ORANGE, AQUA]))
    metrics_ = [("anytime", "anytime accuracy"), ("final", "final accuracy"), ("fewshot", "few-shot score (n <= 256)")]
    fig, axes = plt.subplots(len(metrics_), len(floors), figsize=(5 * len(floors), 3.4 * len(metrics_)),
                             sharey="row", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, g in enumerate(floors):
        for row, (key, label) in enumerate(metrics_):
            ax = axes[row][col]
            style(ax)
            for ae in evicts:
                pts = sorted((r["ar"], r[key]) for r in rows if r["kind"] == "priority" and r["ae"] == ae
                             and r["floor"] == g)
                if pts:
                    ax.plot(*zip(*pts), marker="o", color=colours[ae], lw=2, ms=6,
                            label=f"priority, alpha_evict {ae:g}")
            for r in rows:
                if r["kind"] == "fifo":
                    ls = "-" if r["floor"] == g else ":"
                    ax.axhline(r[key], color=INK2, ls=ls, lw=1.2,
                               label=f"FIFO, gate floor {r['floor']:g}" + ("" if r["floor"] == g else " (other panel)"))
            ax.set_xlabel("alpha_replay (0 = uniform replay)", color=INK2)
            if col == 0:
                ax.set_ylabel(label, color=INK2)
            if row == 0:
                ax.set_title("no gate (store all)" if g == 1 else f"novelty gate, floor {g:g}", color=INK)
    handles, labels = {}, {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            labels.setdefault(l, h)
    fig.legend(list(labels.values()), list(labels.keys()), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("grid_dir")
    p.add_argument("--fig")
    p.add_argument("--title", default="")
    p.add_argument("--prefix", default="", help="only runs named <prefix><config>, e.g. c32_k250_s0__")
    a = p.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.grid_dir, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        if not name.startswith(a.prefix):
            continue
        name = name[len(a.prefix):]
        kind = parse(name)
        if kind is None or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        rows.append({"name": name, "kind": kind[0], "ae": kind[1], "ar": kind[2], "floor": kind[3], **run_metrics(d)})
    rows.sort(key=lambda r: -r["anytime"])
    cols = ["final", "anytime", "worst", "worst_digit", "fewshot", "retain", "written", "digits", "age"]
    print("| run | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in rows:
        print(f"| {r['name']} | " + " | ".join(f"{r[c]:.3f}" if c not in ("age", "digits") else f"{r[c]:.1f}"
                                              for c in cols) + " |")
    if a.fig:
        figure(rows, a.fig, a.title)


if __name__ == "__main__":
    main()
