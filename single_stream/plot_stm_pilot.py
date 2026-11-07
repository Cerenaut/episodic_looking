"""STM pilot (7-8 Nov 2026): the CLS model (LTM + replay + sleep + RL STM) against the LTM alone, over the end of task 1
and task 2, from existing logs. Two figures:

- timeline: held-out accuracy against exposures since the switch (symlog), split into the class of the current
  cluster, the other classes of the current task, and task 1; plus online accuracy (the live sample, before learning);
- few-shot: Memorisation, Generalisation and Protection at each class's n-th exposure in task 2.

    python -m single_stream.plot_stm_pilot --run runs_ss/mnist_clustered_single_stream/stm_pilot/batch2 --out <dir>
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import save, style

CONFIGS = [("lp", 128, 256), ("lp", 128, 1024), ("lp", 512, 256),
           ("dih", 128, 256), ("dih", 128, 1024), ("dih", 512, 256)]
FAMNAME = {"lp": "label perm.", "dih": "dihedral"}
MODELS = [("ltm", "LTM alone", SERIES[0]), ("cls", "LTM + STM, cap 16", SERIES[1]),
          ("cls64", "LTM + STM, cap 64 (stopped early)", SERIES[2])]
NS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def load(run, prefix, fam, c, P):
    d = os.path.join(run, f"{prefix}_{fam}_c{c}_P{P}")
    if not os.path.exists(os.path.join(d, "checkpoints.jsonl")):
        return None
    return {"ckpt": [json.loads(l) for l in open(os.path.join(d, "checkpoints.jsonl"))],
            "fewshot": [json.loads(l) for l in open(os.path.join(d, "fewshot.jsonl"))]}


def series(run_data, task_len):
    """x = exposures since the switch; per checkpoint: current-class, other-class, overall, task 1, online."""
    out = {k: [] for k in ("x", "cur", "other", "acc", "task1", "online")}
    first = run_data["ckpt"][0]["task_name"]
    for r in run_data["ckpt"]:
        x = r["t_task"] - task_len if r["task"] == 0 else r["t_task"]
        age = np.array(r["age"])
        g = np.array(r["group_acc"])
        out["x"].append(x)
        out["cur"].append(g[age == 0].mean() if (age == 0).any() else np.nan)
        out["other"].append(g[age != 0].mean() if (age != 0).any() else np.nan)
        out["acc"].append(r["acc"])
        out["task1"].append(r["task_acc"][first])
        out["online"].append(r["online_acc"] if r["online_acc"] is not None else np.nan)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


ROWS = [("acc", "Held-out accuracy, current task (all classes)"),
        ("cur", "Held-out accuracy, the current cluster's class"),
        ("other", "Held-out accuracy, the other classes"),
        ("task1", "Held-out accuracy, task 1 (id)"),
        ("online", "Online accuracy (live sample, before learning)")]


def fig_timeline(args, data):
    fig, axes = plt.subplots(len(ROWS), len(CONFIGS), figsize=(20, 15), facecolor=SURFACE, sharex=True, sharey=True)
    for j, cfg in enumerate(CONFIGS):
        for i, (key, title) in enumerate(ROWS):
            ax = axes[i, j]
            for prefix, label, colour in MODELS:
                rd = data.get((prefix, *cfg))
                if rd is None:
                    continue
                s = series(rd, 54000)
                keep = s["x"] >= -args.before
                ax.plot(s["x"][keep], s[key][keep], color=colour, lw=1.6, marker="o", ms=2.5, label=label)
            ax.axvline(0, color=INK, lw=0.8, ls=":")
            ax.axvline(-2000, color=SERIES[1], lw=0.8, ls=":")
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(-args.before, 10000)
            ax.set_ylim(0, 1)
            fam, c, P = cfg
            style(ax, f"{FAMNAME[fam]} c {c} P {P}" if i == 0 else "",
                  xlabel="exposures since the switch" if i == len(ROWS) - 1 else "")
            if j == 0:
                ax.set_ylabel(title, fontsize=8.5, color=INK)
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=10, labelcolor=INK, ncol=len(l))
    fig.suptitle("STM pilot batch 2 (seed 0): end of task 1 and task 2. Dotted: the switch (black) and the STM coming "
                 "online (orange, 2,000 before). Sleep (K = P/4) every P exposures; the STM resets at each sleep.",
                 color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, args, "stm_pilot_timeline.png")


FS = [("memorisation", "Memorisation (the class's training samples seen so far)"),
      ("recall", "Generalisation (held-out samples of the class)"),
      ("protection", "Protection (held-out samples of the other classes, current task)")]


def fig_fewshot(args, data):
    fig, axes = plt.subplots(len(FS), len(CONFIGS), figsize=(20, 10), facecolor=SURFACE, sharex=True, sharey=True)
    for j, cfg in enumerate(CONFIGS):
        for i, (key, title) in enumerate(FS):
            ax = axes[i, j]
            for prefix, label, colour in MODELS:
                rd = data.get((prefix, *cfg))
                if rd is None:
                    continue
                ys, ns, cnt = [], [], []
                for n in NS:
                    v = [r[key] for r in rd["fewshot"] if r["task"] == 1 and r["threshold"] == n]
                    if v:
                        ns.append(n)
                        ys.append(np.mean(v))
                        cnt.append(len(v))
                ax.plot(ns, ys, color=colour, lw=1.8, marker="o", ms=4, label=label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS])
            ax.set_ylim(0, 1)
            fam, c, P = cfg
            style(ax, f"{FAMNAME[fam]} c {c} P {P}" if i == 0 else "",
                  xlabel="the class's n-th exposure in task 2" if i == len(FS) - 1 else "")
            if j == 0:
                ax.set_ylabel(title, fontsize=8.5, color=INK)
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=10, labelcolor=INK, ncol=len(l))
    fig.suptitle("STM pilot batch 2 (seed 0): few-shot within task 2, mean over the 10 classes (each scored once at "
                 "its n-th exposure; the first c exposures of a class are its first cluster)",
                 color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save(fig, args, "stm_pilot_fewshot.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--before", type=int, default=54000, help="exposures before the switch to show")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = {}
    for prefix, _, _ in MODELS:
        for cfg in CONFIGS:
            rd = load(args.run, prefix, *cfg)
            if rd is not None:
                data[(prefix, *cfg)] = rd
    print(f"{len(data)} runs")
    fig_timeline(args, data)
    fig_fewshot(args, data)


if __name__ == "__main__":
    main()
