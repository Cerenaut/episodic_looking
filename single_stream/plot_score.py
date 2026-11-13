"""
Four-task summary in the style of report_final's summary figure (plot_final.summary; Dave 13 Nov 2026): one column per
core metric plus the overall score (single_stream/score.py), one row per model variant, each axis zoomed to its own
range, values printed; one block of rows per condition (family x cluster size). Mean over the seeds present; a row
with fewer seeds than the most complete one is labelled with its seed count.

    python -m single_stream.plot_score --run runs_ss/mnist_clustered_single_stream/stm_pilot/four1 --out <dir>
"""
import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, save, style
from single_stream.score import FOUR_TASK, four_task

NAMES = [name for _, name in FOUR_TASK] + ["Overall"]
CONDITIONS = [("lp", 512, "label permutation, c 512"), ("lp", 128, "label permutation, c 128"),
              ("dih", 512, "dihedral, c 512"), ("dih", 128, "dihedral, c 128")]
FOUR1 = [("ltm", "LTM alone", SERIES[0]), ("il0", "STM, no interleave", SERIES[1]),
         ("stop09il0", "STM, stop .9 + no interleave", SERIES[2]), ("il1", "STM, interleave 1 (stage 1 best)", SERIES[3])]
SEEDS = (0, 1)


def rows_for(run, fam, c, models=FOUR1, seeds=SEEDS):
    """[(label, colour, [M, G, P, Overall] seed mean, seeds used)] for the models with at least one finished run."""
    out = []
    for prefix, label, colour in models:
        vals = []
        for s in seeds:
            d = os.path.join(run, f"{prefix}_{fam}_c{c}_s{s}")
            if os.path.exists(os.path.join(d, "summary.json")):
                r = four_task(d)
                vals.append([r[n] for n in NAMES[:-1]] + [r["Score"]])
        if vals:
            out.append((label, colour, np.mean(vals, 0), len(vals)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    blocks = [(title, rows_for(args.run, fam, c)) for fam, c, title in CONDITIONS]
    blocks = [(t, r) for t, r in blocks if r]
    full = max(n for _, rows in blocks for *_, n in rows)
    fig, axes = plt.subplots(len(blocks), len(NAMES), figsize=(16, 1.0 + 0.62 * sum(len(r) + 1 for _, r in blocks)),
                             facecolor=SURFACE, sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [len(r) for _, r in blocks]})
    for i, (title, rows) in enumerate(blocks):
        ys = np.arange(len(rows))[::-1]
        for j, ax in enumerate(axes[i]):
            vals = [v[j] for _, _, v, _ in rows]
            for y, (_, colour, v, _) in zip(ys, rows):
                ax.plot([0, v[j]], [y, y], color=GREY, lw=0.8, zorder=1)
                ax.scatter(v[j], y, s=80, color=colour, zorder=3, edgecolor=SURFACE)
                ax.annotate(f"{v[j]:.2f}", (v[j], y), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=8, color=INK)
            lo, hi = min(vals), max(vals)
            span = hi - lo
            ax.set_xlim(lo - 0.25 * span - 0.01, hi + 0.25 * span + 0.01)
            ax.set_ylim(-0.6, len(rows) - 0.3)
            style(ax, NAMES[j] if i == 0 else "", xlabel="")
        axes[i, 0].set_yticks(ys)
        axes[i, 0].set_yticklabels([label + (f" ({n} seed{'s' if n > 1 else ''})" if n < full else "")
                                    for label, _, _, n in rows])
        axes[i, 0].set_ylabel(title, fontsize=9.5, color=INK)
    fig.suptitle("Four tasks (tasks 2-4 cut to 5,120 exposures; mean of seeds): the core metrics at a class's n-th "
                 "exposure, n = 1-64, and their mean (Overall). Each axis zoomed to its own range.", color=INK,
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "four_task_summary.png")


if __name__ == "__main__":
    main()
