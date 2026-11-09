"""STM pilot batch 3 (8 Nov 2026): few-shot within task 2 (its first 2,048 exposures) for the STM variants against
the LTM alone with the same LTM buffer. A: STM replays from the LTM's priority buffer (500); B: the STM has its own
FIFO buffer (256), LTM priority 500; D: own STM FIFO (256), LTM FIFO 2,048. Cap 256 repeats. The priority-500 LTM
baseline and the cap-16 STM are batch 2's runs (cut at 2,048 exposures).

`--rewards`: the same panels for batch 5 (9 Nov 2026): A with the "what" bias under the p(y) reward and the composite
(accuracy-log, w 1 and .5), against the LTM alone and batch 3's A ("where", p(y)).

    python -m single_stream.plot_stm_batch3 --run runs_ss/mnist_clustered_single_stream/stm_pilot --out <dir> [--rewards]
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_stm_pilot import FAMNAME, FS, NS
from single_stream.plot_wave8 import save, style

CONFIGS = [("lp", 512), ("dih", 512), ("lp", 128)]
MODELS = [  # (directory pattern, label, colour, line style)
    ("batch2/ltm_{f}_c{c}_P256", "LTM alone, priority 500", SERIES[0], "-"),
    ("batch2/cls_{f}_c{c}_P256", "A at cap 16 (batch 2)", SERIES[1], ":"),
    ("batch3/A_{f}_c{c}", "A: STM replays LTM's priority 500", SERIES[1], "-"),
    ("batch3/B_{f}_c{c}", "B: own STM FIFO 256, LTM priority 500", SERIES[2], "-"),
    ("batch3/ltmfifo_{f}_c{c}", "LTM alone, FIFO 2,048", SERIES[0], "--"),
    ("batch3/D_{f}_c{c}", "D: own STM FIFO 256, LTM FIFO 2,048", SERIES[2], "--"),
]
REWARD_MODELS = [
    ("batch2/ltm_{f}_c{c}_P256", "LTM alone, priority 500", SERIES[0], "-"),
    ("batch3/A_{f}_c{c}", "A, \"where\" bias, p(y) reward (batch 3)", SERIES[1], ":"),
    ("batch5/acc_{f}_c{c}", "A, \"what\" bias, p(y) reward", SERIES[1], "-"),
    ("batch5/cw1_{f}_c{c}", "A, \"what\" bias, composite reward w 1", SERIES[2], "-"),
    ("batch5/cw05_{f}_c{c}", "A, \"what\" bias, composite reward w .5", SERIES[2], "--"),
]
MAX_T = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rewards", action="store_true", help="batch 5: p(y) vs composite reward")
    args = ap.parse_args()
    models = REWARD_MODELS if args.rewards else MODELS
    os.makedirs(args.out, exist_ok=True)
    fig, axes = plt.subplots(len(FS), len(CONFIGS), figsize=(16, 10.5), facecolor=SURFACE, sharex=True, sharey=True)
    for j, (fam, c) in enumerate(CONFIGS):
        for pattern, label, colour, ls in models:
            path = os.path.join(args.run, pattern.format(f=fam, c=c), "fewshot.jsonl")
            rows = [r for r in map(json.loads, open(path)) if r["task"] == 1 and r["t_task"] <= MAX_T]
            for i, (key, _) in enumerate(FS):
                ns, ys = [], []
                for n in NS:
                    v = [r[key] for r in rows if r["threshold"] == n]
                    if v:
                        ns.append(n)
                        ys.append(np.mean(v))
                axes[i, j].plot(ns, ys, color=colour, ls=ls, lw=1.8, marker="o", ms=4, label=label)
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
    head = ("STM batch 5 (seed 0), the STM's reward" if args.rewards else "STM pilot batch 3 (seed 0)")
    fig.suptitle(f"{head}: few-shot in task 2's first 2,048 exposures, mean over the classes seen. "
                 "Grey bands: a sleep (LTM update + STM reset)\nfalls between those exposures (sleeps at 16, 272, "
                 "528 ... exposures after the switch; clusters start at multiples of c).",
                 color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, args, "stm_batch5_fewshot.png" if args.rewards else "stm_batch3_fewshot.png")


if __name__ == "__main__":
    main()
