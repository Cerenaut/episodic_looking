"""
Wave 14 figures and tables: the hybrid (STM + gentle continuous LTM learning) against its two ablations, STM only
(w13_adapt256_prio_fstop9) and LTM only (the same LTM setting without the STM), on the 75%-nines stream, 5 seeds,
paired streams and data (--data-seed 0 everywhere; LTM-only seeds 1-4 rerun as wave 14b).

For the hybrid two readings: the system (LTM + STM bias) and the LTM alone (no bias, the ltm_* keys), which shows what
the LTM's own learning contributes and how much it has drifted.

    python -W ignore -m single_stream.plot_wave14 --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, hist, save, style
from single_stream import plot_wave11 as w11

# checkpoints exist after each of the first 64 nines (~sample 85 of 200 on this stream)
NINES = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32, 40, 48, 64)
TICKS = (1, 2, 5, 10, 20, 40, 64)
SEEDS = range(5)
STM_ONLY = "w13_adapt256_prio_fstop9"
SETTINGS = [  # (key, label), gentlest first
    ("o2_lr3e-5", "current input, Adam 3e-5"),
    ("o2_lr1e-4", "current input, Adam 1e-4"),
    ("o2_lr3e-4", "current input, Adam 3e-4"),
    ("o1_rp15_lr1e-4_k1", "+ replay 15, Adam 1e-4"),
    ("o1_rp15_lr1e-4_k4", "+ replay 15, 4 steps Adam 1e-4"),
    ("o1_rp15_head_lr1e-3", "+ replay 15, head only Adam 1e-3"),
]
ABLATION = ("o2_lr3e-5", "o1_rp15_lr1e-4_k4")  # gentlest, and the strongest few-shot gain
KEYS = ("mem_seen", "new_recall", "others_acc")
RUN = None


def names(cfg):
    return [n for n in (f"{cfg}_s{s}" for s in SEEDS) if os.path.exists(os.path.join(RUN, f"{n}.json"))]


def curves(ns, pre=""):
    """Seed mean, (len(NINES), 3): memorised, held-out, digits 0-8; pre "ltm_" reads the hybrid's LTM alone."""
    return np.nanmean([np.stack([w11.at_nines(hist(RUN, n), pre + k, NINES) for k in KEYS], 1) for n in ns], 0)


def stats(ns, pre=""):
    c = curves(ns, pre)
    i = lambda xs: [NINES.index(x) for x in xs]
    H = [hist(RUN, n) for n in ns]
    return {"curves": c, "mem": c[i((1, 2, 5)), 0], "held": c[i((1, 2, 5, 9, 16, 40, 64)), 1],
            "held_mean": c[i(w11.HELD_AT), 1].mean(), "mem_mean": c[i(w11.MEM_AT), 0].mean(),
            # worst over the whole stream (seed mean of each run's minimum), and at 200 samples
            "worst": np.mean([min(e[pre + "others_acc"] for e in h if e.get(pre + "others_acc") is not None) for h in H]),
            "end": np.mean([h[-1][pre + "others_acc"] for h in H]),
            "eps": np.mean([h[-1].get("learning_episodes", np.nan) for h in H])}


def variants(k):
    """(label, names, key prefix, line style) for setting k."""
    return [("hybrid (system)", names(f"w14_hyb_{k}"), "", "-"),
            ("hybrid, LTM alone", names(f"w14_hyb_{k}"), "ltm_", ":"),
            ("LTM only", names(f"w14_ltm_{k}"), "", "--")]


def plot(axes, c, color, ls, label, lw=2, marker=None):
    for j, ax in enumerate(axes):
        ax.plot(NINES, c[:, j], color=color, lw=lw, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1, label=label,
                zorder=3)


def finish(axes, ylim3=(0.84, 0.975)):
    for j, (ax, t) in enumerate(zip(axes, w11.TITLES)):
        style(ax, t, xlabel="nines presented")
        ax.set_xscale("log")
        ax.set_xticks(TICKS)
        ax.set_xticklabels([str(t) for t in TICKS])
        ax.set_ylim(*((0, 1.02) if j < 2 else ylim3))


def fig_hybrid_all(args):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    plot(axes, curves(names(STM_ONLY)), INK, "-", "STM only (first stop .9)", lw=2.8, marker="o")
    for (k, label), color in zip(SETTINGS, SERIES):
        plot(axes, curves(names(f"w14_hyb_{k}")), color, "-", f"hybrid: {label}", lw=1.8)
    finish(axes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.suptitle("Wave 14 (75% nines): the hybrid with six LTM settings against STM only, mean of 5 seeds",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    save(fig, args, "wave14_hybrid_all.png")


def fig_ablation(args):
    fig, axes = plt.subplots(len(ABLATION), 3, figsize=(16, 4.6 * len(ABLATION)), facecolor=SURFACE)
    for row, k in zip(axes, ABLATION):
        label = dict(SETTINGS)[k]
        color = SERIES[[s for s, _ in SETTINGS].index(k)]
        plot(row, curves(names(STM_ONLY)), INK, "-", "STM only", lw=2.4, marker="o")
        for v, ns, pre, ls in variants(k):
            plot(row, curves(ns, pre), color if v != "LTM only" else GREY, ls, v, lw=2.2 if pre == "" else 1.8)
        finish(row, ylim3=(0.88, 0.975))
        row[0].set_title(f"LTM setting: {label}\n" + w11.TITLES[0], color=INK, loc="left", fontsize=10.5)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.suptitle("Ablation: hybrid vs its parts, mean of 5 seeds (same streams and data)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, args, "wave14_ablation.png")


def fig_tradeoff(args):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    s0 = stats(names(STM_ONLY))
    for ax, key in zip(axes, ("held_mean", "mem_mean")):
        for (k, label), color in zip(SETTINGS, SERIES):
            h, l = stats(names(f"w14_hyb_{k}")), stats(names(f"w14_ltm_{k}"))
            ax.annotate("", (h["worst"], h[key]), (l["worst"], l[key]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.7), zorder=2)
            ax.scatter(l["worst"], l[key], s=50, facecolor=SURFACE, edgecolor=color, lw=1.5, zorder=3)
            ax.scatter(h["worst"], h[key], s=60, color=color, edgecolor=SURFACE, lw=0.8, zorder=4, label=label)
        ax.scatter(s0["worst"], s0[key], s=170, marker="*", color=INK, zorder=5, label="STM only")
        style(ax, "Few-shot: held-out recall, mean at 1, 2, 5, 9 nines" if key == "held_mean"
              else "Memorised: recall on the nines shown so far, mean at 1, 2, 5 nines",
              xlabel="protection: worst accuracy on digits 0-8 over the whole stream")
        ax.set_ylim(0, 1.02)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Each LTM setting alone (open circle) and in the hybrid (filled), arrow from LTM only to hybrid; "
                 "up and right is better", color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, args, "wave14_tradeoff.png")


def table(args):
    f = lambda v: " / ".join(f"{x:.2f}" for x in v)
    L = ["## Wave 14 (75% nines, mean of 5 seeds; values right after the n-th nine)", "",
         "| setting | variant | memorised 1/2/5 | held-out 1/2/5/9/16/40/64 | digits 0-8 worst (whole stream) / at 200 "
         "| learning episodes |", "|---|---|---|---|---|---|"]
    s = stats(names(STM_ONLY))
    L.append(f"| - | STM only | {f(s['mem'])} | {f(s['held'])} | {s['worst']:.3f} / {s['end']:.3f} | {s['eps']:,.0f} |")
    for k, label in SETTINGS:
        for v, ns, pre, _ in variants(k):
            s = stats(ns, pre)
            eps = "-" if np.isnan(s["eps"]) or v == "LTM only" else f"{s['eps']:,.0f}"
            L.append(f"| {label} | {v} | {f(s['mem'])} | {f(s['held'])} | {s['worst']:.3f} / {s['end']:.3f} | {eps} |")
    L += ["", "## Pre-registered test (wave 12): A beats B if memorised >= at 1, 2, 5 nines, held-out >= at 1, 2, 5, "
          "9 nines, digits 0-8 >= (within -0.005) at every checkpoint; seed means, paired. Margins: worst A - B "
          "(memorised / held-out / digits 0-8)", "",
          "| setting | hybrid vs STM only | hybrid vs LTM only | LTM only vs STM only |", "|---|---|---|---|"]
    cell = lambda r: f"{'**yes**' if r[0] else 'no'} ({r[1]:+.2f} / {r[2]:+.2f} / {r[3]:+.3f})"
    for k, label in SETTINGS:
        hy, lt = f"w14_hyb_{k}", f"w14_ltm_{k}"
        L.append(f"| {label} | {cell(w11.dominance(hy, STM_ONLY))} | {cell(w11.dominance(hy, lt))} | "
                 f"{cell(w11.dominance(lt, STM_ONLY))} |")
    L += ["", "## Paired held-out difference, hybrid - STM only (mean +- SE over seeds) at 1/2/5/9/16/40 nines", "",
          "| setting | difference |", "|---|---|"]
    G = (1, 2, 5, 9, 16, 40)
    for k, label in SETTINGS:
        d = np.array([w11.at_nines(hist(RUN, f"w14_hyb_{k}_s{x}"), "new_recall", G)
                      - w11.at_nines(hist(RUN, f"{STM_ONLY}_s{x}"), "new_recall", G) for x in SEEDS])
        m, se = d.mean(0), d.std(0, ddof=1) / np.sqrt(len(SEEDS))
        L.append(f"| {label} | " + " / ".join(f"{a:+.2f}+-{b:.2f}" for a, b in zip(m, se)) + " |")
    path = os.path.join(args.out, "wave14_table.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(path)


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RUN = w11.RUN = args.run
    w11.SEEDS = SEEDS
    os.makedirs(args.out, exist_ok=True)
    fig_hybrid_all(args)
    fig_ablation(args)
    fig_tradeoff(args)
    table(args)


if __name__ == "__main__":
    main()
