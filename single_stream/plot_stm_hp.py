"""STM hyperparameter search (10 Nov 2026): few-shot within task 2 (its first 2,048 exposures) for each setting,
mean over seeds 0 and 1, against the LTM alone, laid out as the batch 3 figure (plot_stm_batch3). Stage 1: around
batch 5's composite reward w 1 (log weight w 2, 4; cap 512). Stage 2: around stage 1's winner, w 1
cap 512 (2a: stop threshold, exploration, STM learning rate; 2b: interleaved replay, w 2). Stage 3: the four
contenders (stage 1 best, interleave 0, stop .9, both) at seeds 0, 1 and 2.

    python -m single_stream.plot_stm_hp --run runs_ss/mnist_clustered_single_stream/stm_pilot --stage 1 --out <dir>
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, INK2, SERIES, SURFACE
from single_stream.plot_stm_pilot import FAMNAME, FS, NS
from single_stream.plot_wave8 import save, style

CONFIGS = [("lp", 512), ("dih", 512)]
LTM = ({0: "batch2/ltm_{f}_c{c}_P256", 1: "hp1/ltm_{f}_c{c}_s1"}, "LTM alone", SERIES[0], "-")
STAGES = {  # setting: ({seed: directory pattern}, label, colour, line style)
    1: [LTM,
        ({0: "batch5/acc_{f}_c{c}"}, "STM, p(y) reward (seed 0 only)", SERIES[1], ":"),
        ({0: "batch5/cw1_{f}_c{c}", 1: "hp1/w1_{f}_c{c}_s1"}, "STM, composite w 1, cap 256", SERIES[2], "--"),
        ({s: f"hp1/w2_{{f}}_c{{c}}_s{s}" for s in (0, 1)}, "STM, composite w 2, cap 256", SERIES[3], "-"),
        ({s: f"hp1/w4_{{f}}_c{{c}}_s{s}" for s in (0, 1)}, "STM, composite w 4, cap 256", SERIES[4], "-"),
        ({s: f"hp1/cap512_{{f}}_c{{c}}_s{s}" for s in (0, 1)}, "STM, composite w 1, cap 512", SERIES[5], "-")],
}
CAP512 = ({s: f"hp1/cap512_{{f}}_c{{c}}_s{s}" for s in (0, 1)}, "STM, stage 1 best (w 1, cap 512)", INK2, "--")
HP2 = lambda arm, label, colour: ({s: f"hp2/{arm}_{{f}}_c{{c}}_s{s}" for s in (0, 1)}, label, colour, "-")
STAGES["2a"] = [LTM, CAP512, HP2("stop09", "stop at p(y) .9", SERIES[1]), HP2("std025", "policy std .25", SERIES[2]),
                HP2("std1", "policy std 1", SERIES[3]), HP2("lr0005", "STM lr .005", SERIES[4]),
                HP2("lr002", "STM lr .02", SERIES[5])]
STAGES["2b"] = [LTM, CAP512, HP2("il0", "no interleaved replay", SERIES[1]),
                HP2("il2", "2 interleaved replay episodes", SERIES[2]), HP2("w2", "log weight w 2", SERIES[3])]
S3 = lambda arm, d, label, colour, ls="-": ({0: f"{d}/{arm}_{{f}}_c{{c}}_s0", 1: f"{d}/{arm}_{{f}}_c{{c}}_s1",
                                              2: f"seed2/{arm}_{{f}}_c{{c}}_s2"}, label, colour, ls)
STAGES[3] = [({**LTM[0], 2: "seed2/ltm_{f}_c{c}_s2"}, *LTM[1:]),
             S3("cap512", "hp1", "STM, stage 1 best (w 1, cap 512)", INK2, "--"),
             S3("il0", "hp2", "no interleaved replay", SERIES[1]), S3("stop09", "hp2", "stop at p(y) .9", SERIES[2]),
             S3("stop09il0", "hp3", "stop at p(y) .9 + no interleaved replay", SERIES[3])]
STAGES = {str(k): v for k, v in STAGES.items()}
MAX_T = 2048


def curve(run, patterns, fam, c, key):
    """Mean over seeds of the per-seed mean over classes at each n; None if a run is missing."""
    per_seed = []
    for pattern in patterns.values():
        path = os.path.join(run, pattern.format(f=fam, c=c), "fewshot.jsonl")
        if not os.path.exists(path):
            return None
        rows = [r for r in map(json.loads, open(path)) if r["task"] == 1 and r["t_task"] <= MAX_T]
        per_seed.append([np.mean([r[key] for r in rows if r["threshold"] == n]) for n in NS])
    return np.mean(per_seed, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(len(FS), len(CONFIGS), figsize=(12, 10.5), facecolor=SURFACE, sharex=True, sharey=True)
    for j, (fam, c) in enumerate(CONFIGS):
        for patterns, label, colour, ls in STAGES[args.stage]:
            for i, (key, _) in enumerate(FS):
                ys = curve(args.run, patterns, fam, c, key)
                if ys is not None:
                    axes[i, j].plot(NS, ys, color=colour, ls=ls, lw=1.8, marker="o", ms=4, label=label)
        for i, (_, title) in enumerate(FS):
            ax = axes[i, j]
            for x in (12, 384):  # a sleep falls between n = 8 and 16, and between 256 and 512
                ax.axvspan(x * 0.95, x * 1.05, color=INK, alpha=0.12, lw=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS])
            ax.set_ylim(0, 1)
            style(ax, f"{FAMNAME[fam]} c {c} P 256" if i == 0 else "",
                  xlabel="the class's n-th exposure in task 2" if i == len(FS) - 1 else "")
            if j == 0:
                ax.set_ylabel(title.replace(" (", "\n("), fontsize=8.5, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=10, labelcolor=INK, ncol=3)
    seeds = [str(k) for k in max((p for p, *_ in STAGES[args.stage]), key=len)]
    seeds = ", ".join(seeds[:-1]) + " and " + seeds[-1]
    fig.suptitle(f"STM hyperparameter search, stage {args.stage} (\"what\" bias; mean of seeds {seeds}): few-shot in "
                 "task 2's first 2,048 exposures,\nmean over the classes seen. Grey bands: a sleep (LTM update + STM "
                 "reset) falls between those exposures.", color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save(fig, args, f"stm_hp{args.stage}_fewshot.png")


if __name__ == "__main__":
    main()
