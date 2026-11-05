"""
Wave 16b (experiment B): a second phase with digit 7 at 75% and no nines, after phase 1 (75% nines, cut after the 64th
nine). Sleep, if any, once between the phases. Reads 7 bounce-back (mem2_seen, new2_recall: held-out 7s), 9 retention
(mem_seen, new_recall) and the digits other than 7 and 9 (others2_acc).

    python -W ignore -m single_stream.plot_wave16b --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, save, style

RUN = None
SEEDS = range(5)
P1 = 64  # nines in phase 1
SEVENS = list(range(1, 65))
END = 153  # x for the end of phase 2 (mean 7s presented by sample 200)
# (config, label, colour, line style); colour follows the model in every figure; dashed = the gentler variant
CONFIGS = [("stm_none", "STM only, no sleep", INK, "-"),
           ("stm_gentle_reset", "STM only, gentle sleep + reset", SERIES[0], "-"),
           ("stm_middle_reset", "STM only, middle sleep + reset", SERIES[1], "-"),
           ("ltm_3e-5_none", "LTM gentlest, no sleep", SERIES[2], "--"),
           ("ltm_3e-5_gentle", "LTM gentlest, gentle sleep", SERIES[2], "-"),
           ("ltm_k4_none", "LTM balanced, no sleep", SERIES[3], "-"),
           ("hyb_3e-5_none", "hybrid, gentlest LTM", SERIES[4], "--"),
           ("hyb_k4_none", "hybrid, balanced LTM", SERIES[4], "-")]
STYLE = {c: (label, col, ls) for c, label, col, ls in CONFIGS}
KEYS = ("mem2_seen", "new2_recall", "others2_acc", "mem_seen", "new_recall")
TITLES = ("7: memorised (shown-so-far 7 shots)", "7: held-out 7s", "digits other than 7 and 9",
          "9: memorised (the 16 shots)", "9: held-out 9s (retention)")


def hists(cfg):
    return [json.load(open(os.path.join(RUN, f"w16b_{cfg}_s{s}.json")))["result"]["history"] for s in SEEDS]


val = lambda e, k: np.nan if e.get(k) is None else e[k]


def phase2(H):
    """x (7s presented; 0 twice when there is a sleep: before and after), y (n, len(KEYS)); the end of phase 2 at END."""
    xs, ys = [], []
    for after in (False, True):
        es = [[e for e in h if e.get("new_samples") == P1 and not e.get("new2_samples") and bool(e.get("after_sleep"))
               == after] for h in H]
        if all(es):
            xs.append(0)
            ys.append(np.nanmean([[val(e[-1], k) for k in KEYS] for e in es], 0))
    for n in SEVENS:
        xs.append(n)
        ys.append(np.nanmean([[val([e for e in h if e.get("new2_samples") == n][-1], k) for k in KEYS] for h in H], 0))
    xs.append(END)
    ys.append(np.nanmean([[val(h[-1], k) for k in KEYS] for h in H], 0))
    return np.array(xs), np.array(ys)


def stream(H):
    """Both phases on one axis: phase 1 in nines (0..64), then phase 2 in 7s (64 + 7s presented)."""
    xs, ys = [], []
    for n in range(1, P1 + 1):
        xs.append(n)
        ys.append(np.nanmean([[val([e for e in h if e.get("new_samples") == n and not e.get("new2_samples")
                                    and not e.get("after_sleep")][-1], k) for k in KEYS] for h in H], 0))
    x2, y2 = phase2(H)
    return np.concatenate([xs, P1 + x2]), np.concatenate([ys, y2])


def fig_stream(args):
    cfgs = ["stm_none", "stm_middle_reset", "ltm_3e-5_gentle", "ltm_k4_none", "hyb_k4_none"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for c in cfgs:
        label, col, ls = STYLE[c]
        x, y = stream(hists(c))
        for ax, j in zip(axes, (4, 1, 2)):
            ax.plot(x, y[:, j], color=col, ls=ls, lw=2.4 if c == "stm_middle_reset" else 1.8, label=label, zorder=3)
    for ax, t in zip(axes, ("held-out 9s", "held-out 7s", "digits other than 7 and 9")):
        style(ax, t, xlabel="phase 1: nines presented   |   phase 2: 7s presented")
        ax.axvline(P1, color=GREY, lw=1, ls=":", zorder=1)
        ax.set_xticks([0, 32, 64, 64 + 32, 64 + 64, 64 + END])
        ax.set_xticklabels(["0", "32", "64 | 0", "32", "64", "end"])
        ax.text(P1 / 2, 0.02, "phase 1:\n75% nines", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=GREY, fontsize=8.5)
        ax.text(P1 + END / 2, 0.02, "phase 2: 75% 7s, no nines", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", color=GREY, fontsize=8.5)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0.6, 1.0)
    axes[2].set_ylim(0.9, 0.98)
    axes[0].legend(loc="center right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Both phases: the nines are learned, then the 7s (damaged in phase 1) bounce back (mean of 5 seeds)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args, "wave16b_stream.png")


def fig_phase2(args, cfgs, name, title):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURFACE)
    axs = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for c in cfgs:
        label, col, ls = STYLE[c]
        x, y = phase2(hists(c))
        for j, ax in enumerate(axs):
            ax.plot(x, y[:, j], color=col, ls=ls, lw=2.4 if c == "stm_middle_reset" else 1.8, label=label, zorder=3)
    for j, ax in enumerate(axs):
        style(ax, TITLES[j], xlabel="7s presented in phase 2 (0 = end of phase 1)")
        ax.set_xscale("symlog", linthresh=1)
        t = [0, 1, 2, 5, 10, 20, 64, END]
        ax.set_xticks(t)
        ax.set_xticklabels([str(v) for v in t[:-1]] + ["end"])
        ax.set_xlim(-0.1, END * 1.15)
    for ax, lim in zip(axs, [(0.6, 1.02), (0.75, 1.0), (0.9, 0.98), (0.6, 1.02), (0.5, 1.0)]):
        ax.set_ylim(*lim)
    axes[1, 2].axis("off")
    h, l = axs[0].get_legend_handles_labels()
    axes[1, 2].legend(h, l, loc="center", frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, name)


def table(args):
    L = ["## Wave 16b (experiment B): phase 2 = digit 7 at 75%, no nines (mean of 5 seeds)", "",
         "P1 end = end of phase 1 after the sleep, if any. 7s: held-out 7s (before phase 1: .99). Other: digits other "
         "than 7 and 9.", "",
         "| config | 7 held-out: P1 end / 1 / 5 / 16 / end | 7 memorised at 1 / 2 / 5 | 9 memorised: P1 end / end | "
         "9 held-out: P1 end / worst / end | other: P1 end / worst / end |", "|---|---|---|---|---|---|"]
    for c, label, _, _ in CONFIGS:
        H = hists(c)
        x, y = phase2(H)
        i = lambda n: np.nonzero(x == n)[0][-1]
        f = lambda v: "/".join(f"{a:.2f}" for a in v)
        p2 = [[e for e in h if e.get("new2_samples", 0) > 0] for h in H]
        w9 = np.mean([min(e["new_recall"] for e in es) for es in p2])
        wo = np.mean([min(e["others2_acc"] for e in es) for es in p2])
        L.append(f"| {label} | {f(y[[i(0), i(1), i(5), i(16), i(END)], 1])} | {f(y[[i(1), i(2), i(5)], 0])} | "
                 f"{f(y[[i(0), i(END)], 3])} | {y[i(0), 4]:.2f} / {w9:.2f} / {y[i(END), 4]:.2f} | "
                 f"{y[i(0), 2]:.3f} / {wo:.3f} / {y[i(END), 2]:.3f} |")
    path = os.path.join(args.out, "wave16b_table.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(path)


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RUN = args.run
    os.makedirs(args.out, exist_ok=True)
    fig_stream(args)
    fig_phase2(args, ["stm_none", "stm_gentle_reset", "stm_middle_reset"], "wave16b_stm.png",
               "Phase 2, STM only: no sleep vs a sleep between the phases + STM reset (mean of 5 seeds)")
    fig_phase2(args, ["stm_middle_reset", "ltm_3e-5_none", "ltm_3e-5_gentle", "ltm_k4_none", "hyb_3e-5_none",
                      "hyb_k4_none"], "wave16b_ltm_hybrid.png",
               "Phase 2, LTM only and hybrids, against STM only + middle sleep + reset (mean of 5 seeds)")
    table(args)


if __name__ == "__main__":
    main()
