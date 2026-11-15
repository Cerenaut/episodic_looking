"""
Four-task summary in the style of report_final's summary figure (plot_final.summary; Dave 13 Nov 2026): one column per
core metric plus the overall score (single_stream/score.py), one row per model variant, each axis zoomed to its own
range, values printed; one block of rows per condition (family x cluster size). Mean over the seeds present; a row
with fewer seeds than the most complete one is labelled with its seed count.
--set cmp (Dave 14 Nov 2026) adds the comparison models (comparisons/cmp4) to four1's LTM alone and RL STM, and two
more figures: the core metrics against n (tasks 2-4 pooled). Sleeps (every 256 exposures from the start of the
stream) are phase-locked to the clusters: task 2 starts at 54,000 = 240 mod 256, so at c 512 every class's 16th
exposure comes right after a sleep (c 128: half of them), hence a plot against exposures since sleep confounds n.

    python -m single_stream.plot_score --root runs_ss/mnist_clustered_single_stream --set four1 --out <dir>
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, INK2, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, save, style
from single_stream.score import FOUR_TASK, POINTS, four_task

NAMES = [name for _, name in FOUR_TASK] + ["Overall"]
CONDITIONS = [("lp", 512, "label permutation, c 512"), ("lp", 128, "label permutation, c 128"),
              ("dih", 512, "dihedral, c 512"), ("dih", 128, "dihedral, c 128")]
# (directory under --root, run prefix, label, colour, line style, sleeps)
FOUR1 = [("stm_pilot/four1", "ltm", "LTM alone", SERIES[0], "-", True),
         ("stm_pilot/four1", "il0", "STM, no interleave", SERIES[1], "-", True),
         ("stm_pilot/four1", "stop09il0", "STM, stop .9 + no interleave", SERIES[2], "-", True),
         ("stm_pilot/four1", "il1", "STM, interleave 1 (stage 1 best)", SERIES[3], "-", True)]
CMP = "comparisons/cmp4"
CMP4 = [(CMP, "ladder1", "LTM, each sample (ladder 1)", INK2, ":", False),
        (CMP, "ladder2", "LTM + replay, no sleep (ladder 2)", GREY, "--", False),
        (CMP, "er", "ER (ladder 2 + live sample)", INK, "-.", False),
        ("stm_pilot/four1", "ltm", "LTM + replay + sleep (LTM alone)", SERIES[0], "-", True),
        ("stm_pilot/four1", "il0", "LTM + RL STM (no interleave)", SERIES[1], "-", True)] + [
        (CMP, f"{h}_{r}", f"{name} head, {'reset' if r == 'reset' else 'kept'}", col, "-" if r == "reset" else "--", True)
        for h, name, col in (("knn", "kNN", SERIES[2]), ("fly", "FlyModel", SERIES[3]), ("sdmlp", "SDMLP", SERIES[4]))
        for r in ("reset", "keep")]
# Head tuning (Dave 15 Nov 2026), stage 2 winners: runs in tune2, seed 0 at c 512 in tune1/tune1b (a tuple of
# directories: the first holding the run is used). Scored on task 2's first 2,048 exposures (WINDOWS), so four1's RL
# STM and LTM alone are scored on the same window as the two-task tuning runs.
TUNE = ("comparisons/tune2", "comparisons/tune1", "comparisons/tune1b")
TUNED = [("stm_pilot/four1", "ltm", "LTM + replay + sleep (LTM alone)", SERIES[0], "-", True),
         ("stm_pilot/four1", "il0", "LTM + RL STM (no interleave)", SERIES[1], "-", True),
         (TUNE, "knn_default_reset", "kNN head, reset (defaults)", SERIES[2], "-", True),
         (TUNE, "knn_replay0_keep", "kNN head, kept (replay 0)", SERIES[2], "--", True),
         (TUNE, "fly_lr2_reset", "FlyModel head, reset (lr 2)", SERIES[3], "-", True),
         (TUNE, "fly_lr2_keep", "FlyModel head, kept (lr 2)", SERIES[3], "--", True),
         (TUNE, "sdmlp_rep64_lr0.5_reset", "SDMLP head, reset (repeats 64, lr .5)", SERIES[4], "-", True),
         (TUNE, "sdmlp_rep64_lr0.5_keep", "SDMLP head, kept (repeats 64, lr .5)", SERIES[4], "--", True)]
SETS = {"four1": FOUR1, "cmp": CMP4, "tune": TUNED}
SET_SEEDS = {"tune": (0, 1, 2)}
WINDOWS = {"tune": {"last_task": 1, "max_t_task": 2048}}
SEEDS = (0, 1)
WINDOW = {}
# (summary title, curves title opening) when not the four-task defaults
TITLES = {"tune": ("Head tuning, stage 2 winners: task 2's first 2,048 exposures (heads: seeds 0-2; LTM alone and RL STM: "
                   "four1, seeds 0-1). Core metrics at a class's n-th exposure, n = 1-64, and Overall.",
                   "Head tuning, stage 2 winners: the core metrics at a class's n-th exposure in task 2 (first 2,048 "
                   "exposures).")}


def run_dirs(root, model, fam, c, seeds=None):
    dirs, prefix = model[:2]
    dirs = (dirs,) if isinstance(dirs, str) else dirs
    out = []
    for s in seeds or SEEDS:
        found = [x for x in (os.path.join(root, d, f"{prefix}_{fam}_c{c}_s{s}") for d in dirs)
                 if os.path.exists(os.path.join(x, "summary.json"))]
        out += found[:1]
    return out


def rows_for(root, fam, c, models=FOUR1, seeds=None):
    """[(label, colour, [M, G, P, Overall] seed mean, seeds used, line style)] for the models with at least one finished
    run."""
    out = []
    for model in models:
        vals = []
        for d in run_dirs(root, model, fam, c, seeds):
            r = four_task(d, **WINDOW)
            vals.append([r[n] for n in NAMES[:-1]] + [r["Score"]])
        if vals:
            out.append((model[2], model[3], np.mean(vals, 0), len(vals), model[4]))
    return out


def records(d, first_task=1):
    last, max_t = WINDOW.get("last_task"), WINDOW.get("max_t_task")
    return [r for r in map(json.loads, open(os.path.join(d, "fewshot.jsonl"))) if r["task"] >= first_task
            and (last is None or r["task"] <= last) and (max_t is None or r["t_task"] <= max_t)]


def fig_curves(args, models):
    """The core metrics at a class's n-th exposure (tasks 2-4 pooled), mean over seeds of each seed's mean."""
    fig, axes = plt.subplots(len(FOUR_TASK), len(CONDITIONS), figsize=(16, 10.5), facecolor=SURFACE, sharex=True,
                             sharey="row")
    for j, (fam, c, title) in enumerate(CONDITIONS):
        for model in models:
            runs = run_dirs(args.root, model, fam, c)
            if not runs:
                continue
            R = [records(d) for d in runs]
            for i, (key, _) in enumerate(FOUR_TASK):
                ys = np.mean([[np.mean([r[key] for r in rs if r["threshold"] == n]) for n in POINTS] for rs in R], 0)
                axes[i, j].plot(POINTS, ys, color=model[3], ls=model[4], lw=1.8, marker="o", ms=3.5, label=model[2])
        for i, (_, name) in enumerate(FOUR_TASK):
            ax = axes[i, j]
            ax.axvspan(12 * 0.95, 12 * 1.05, color=INK, alpha=0.12, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(POINTS)
            ax.set_xticklabels([str(n) for n in POINTS])
            style(ax, title if i == 0 else "", xlabel="the class's n-th exposure in its task" if i == 2 else "")
            if j == 0:
                ax.set_ylabel(name, fontsize=10, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=9.5, labelcolor=INK, ncol=4)
    head = (TITLES.get(args.set, ("", ""))[1] or "Four tasks: the core metrics at a class's n-th exposure, tasks 2-4 pooled "
            "(mean of seeds 0 and 1).")
    fig.suptitle(head + " Grey band: a sleep falls between n = 8 and 16\n(every cluster at c 512, half of them at c 128); a reset head "
                 "is empty at n = 16 and the LTM answers.",
                 color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    save(fig, args, "four_task_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="runs_ss/mnist_clustered_single_stream")
    ap.add_argument("--set", default="four1", choices=list(SETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global SEEDS, WINDOW
    models = SETS[args.set]
    SEEDS, WINDOW = SET_SEEDS.get(args.set, SEEDS), WINDOWS.get(args.set, {})
    blocks = [(title, rows_for(args.root, fam, c, models)) for fam, c, title in CONDITIONS]
    blocks = [(t, r) for t, r in blocks if r]
    full = max(r[3] for _, rows in blocks for r in rows)
    fig, axes = plt.subplots(len(blocks), len(NAMES), figsize=(16, 1.0 + 0.62 * sum(len(r) + 1 for _, r in blocks)),
                             facecolor=SURFACE, sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [len(r) for _, r in blocks]})
    for i, (title, rows) in enumerate(blocks):
        ys = np.arange(len(rows))[::-1]
        for j, ax in enumerate(axes[i]):
            vals = [r[2][j] for r in rows]
            for y, (_, colour, v, _, ls) in zip(ys, rows):
                ax.plot([0, v[j]], [y, y], color=GREY, lw=0.8, zorder=1)
                ax.scatter(v[j], y, s=80, color=colour, marker="D" if ls == "--" else "o", zorder=3, edgecolor=SURFACE)
                ax.annotate(f"{v[j]:.2f}", (v[j], y), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=8, color=INK)
            lo, hi = min(vals), max(vals)
            span = hi - lo
            ax.set_xlim(lo - 0.25 * span - 0.01, hi + 0.25 * span + 0.01)
            ax.set_ylim(-0.6, len(rows) - 0.3)
            style(ax, NAMES[j] if i == 0 else "", xlabel="")
        axes[i, 0].set_yticks(ys)
        axes[i, 0].set_yticklabels([label + (f" ({n} seed{'s' if n > 1 else ''})" if n < full else "")
                                    for label, _, _, n, _ in rows])
        axes[i, 0].set_ylabel(title, fontsize=9.5, color=INK)
    fig.suptitle(TITLES.get(args.set, ("", ""))[0] or "Four tasks (tasks 2-4 cut to 5,120 exposures; mean of seeds): the core "
                 "metrics at a class's n-th exposure, n = 1-64, and their mean (Overall). Each axis zoomed to its own range.",
                 color=INK,
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "four_task_summary.png")
    if args.set != "four1":
        fig_curves(args, models)


if __name__ == "__main__":
    main()
