"""
Wave 15: more STM replay on the latest LTM (--replay-per-sample 64 / 128, and 64 uniform) for STM only and two
hybrids, plus the false-"9" diagnosis (others_as_new: digits 0-8 answered 9) for the three model families. Also writes
the three-model summary table (STM only, LTM only, hybrid) used in the wrap-up report.

    python -W ignore -m single_stream.plot_wave15 --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
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
from single_stream import plot_wave14 as w14

SETTINGS = [("o2_lr3e-5", "LTM current input, Adam 3e-5"), ("o1_rp15_lr1e-4_k4", "LTM + replay 15, 4 steps Adam 1e-4")]
REPLAY = [("base", "adaptive (as wave 14)"), ("rps64", "64 replays per sample"), ("rps128", "128 replays per sample"),
          ("u64", "64 per sample, uniform")]
STM_REPLAY = [("base", "adaptive (as wave 13)"), ("rps64", "64 replays per sample"),
              ("rps128", "128 replays per sample")]
# representatives of the three families for the summary (all 75% nines, 5 seeds, data seed 0)
SUMMARY = [("STM only", "STM: adapt 256 + prio + first stop .9", "w15_stm_base"),
           ("LTM only", "fast: Adam 1e-3, every sample", "w13_ltm_every1"),
           ("LTM only", "balanced: replay 15, 4 steps Adam 1e-4", "w15_ltm_o1_rp15_lr1e-4_k4"),
           ("LTM only", "gentlest: current input, Adam 3e-5", "w15_ltm_o2_lr3e-5"),
           ("hybrid", "STM + balanced LTM", "w15_hyb_o1_rp15_lr1e-4_k4_base"),
           ("hybrid", "STM + gentlest LTM", "w15_hyb_o2_lr3e-5_base")]


def damage(ns):
    """At each run's worst checkpoint on digits 0-8: the drop from the start, the part that is a false "9" (system and
    LTM alone), and the per-digit drop; seed means."""
    R = []
    for n in ns:
        h = [e for e in hist(w14.RUN, n) if e.get("others_acc") is not None]
        e0, e = h[0], min(h, key=lambda e: e["others_acc"])
        R.append([e0["others_acc"] - e["others_acc"], e.get("others_as_new", np.nan) - e0.get("others_as_new", np.nan),
                  e.get("ltm_others_as_new", np.nan) - e0.get("ltm_others_as_new", np.nan) if "ltm_others_as_new" in e
                  else np.nan] + list(np.array(e0["per_digit"][:9]) - np.array(e["per_digit"][:9])))
    return np.nanmean(R, 0)


def false9_curve(ns, pre=""):
    return np.nanmean([w11.at_nines(hist(w14.RUN, n), pre + "others_as_new", w14.NINES) for n in ns], 0)


def fig_false9(args):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURFACE)
    for ax, (k, label) in zip(axes, SETTINGS):
        ax.plot(w14.NINES, false9_curve(w14.names("w15_stm_base")), color=INK, lw=2.4, marker="o", ms=4,
                label="STM only")
        ax.plot(w14.NINES, false9_curve(w14.names(f"w15_ltm_{k}")), color=GREY, lw=2, ls="--", label="LTM only")
        hy = w14.names(f"w15_hyb_{k}_base")
        ax.plot(w14.NINES, false9_curve(hy), color=SERIES[0], lw=2.4, label="hybrid (system)")
        ax.plot(w14.NINES, false9_curve(hy, "ltm_"), color=SERIES[0], lw=1.6, ls=":", label="hybrid, LTM alone")
        style(ax, f"{label}\ndigits 0-8 answered \"9\" (share of digits 0-8 test images)", xlabel="nines presented")
        ax.set_xscale("log")
        ax.set_xticks(w14.TICKS)
        ax.set_xticklabels([str(t) for t in w14.TICKS])
        ax.set_ylim(0, None)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("False \"9\"s: the hybrid makes more than its two parts together (mean of 5 seeds, 75% nines)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave15_false9.png")


def row(name, ns):
    s = w14.stats(ns)
    d = damage(ns)
    f = lambda v: "/".join(f"{x:.2f}" for x in v)
    top = np.argsort(-d[3:])[:2]
    return (f"| {name} | {f(s['mem'])} | {f(s['held'][[0, 1, 2, 3, 5]])} | {s['worst']:.3f} / {s['end']:.3f} | "
            f"{d[0]:.3f} / " + ("-" if np.isnan(d[1]) else f"{d[1]:.3f}") + " | " + ", ".join(f"{t}: {d[3 + t]:.2f}" for t in top) + " | "
            + ("-" if np.isnan(s["eps"]) else f"{s['eps']:,.0f}") + " |")


def table(args):
    H = ("| config | memorised 1/2/5 | held-out 1/2/5/9/40 | digits 0-8 worst (whole stream) / at 200 | "
         "at the worst point: drop / of which false \"9\" | digits most hurt (drop) | learning episodes |")
    L = ["## Wave 15: more STM replay (75% nines, mean of 5 seeds)", "", H, "|---|---|---|---|---|---|---|"]
    for v, label in STM_REPLAY:
        L.append(row(f"STM only, {label}", w14.names(f"w15_stm_{v}")))
    for k, klabel in SETTINGS:
        L.append(row(f"LTM only ({klabel})", w14.names(f"w15_ltm_{k}")))
        for v, label in REPLAY:
            L.append(row(f"hybrid ({klabel}), STM {label}", w14.names(f"w15_hyb_{k}_{v}")))
    L += ["", "## Pre-registered test vs STM only (adaptive) and vs STM only with the same replay; margins: worst A - B "
          "(memorised / held-out / digits 0-8)", "", "| hybrid | vs STM only | vs STM only, same replay |", "|---|---|---|"]
    cell = lambda r: f"{'**yes**' if r[0] else 'no'} ({r[1]:+.2f} / {r[2]:+.2f} / {r[3]:+.3f})"
    for k, klabel in SETTINGS:
        for v, label in REPLAY:
            same = f"w15_stm_{v}" if v != "u64" else "w15_stm_rps64"
            L.append(f"| {klabel}, STM {label} | {cell(w11.dominance(f'w15_hyb_{k}_{v}', 'w15_stm_base'))} | "
                     f"{cell(w11.dominance(f'w15_hyb_{k}_{v}', same))} |")
    L += ["", "## Hybrid's LTM alone: false \"9\" at the hybrid's worst point", "", "| hybrid | system | LTM alone |",
          "|---|---|---|"]
    for k, klabel in SETTINGS:
        d = damage(w14.names(f"w15_hyb_{k}_base"))
        L.append(f"| {klabel} | {d[1]:.3f} | {d[2]:.3f} |")
    L += ["", "## Three-model summary (representatives; 75% nines, 5 seeds, data seed 0)", "", H.replace("config", "family | config"),
          "|---|---|---|---|---|---|---|---|"]
    for fam, label, cfg in SUMMARY:
        L.append(row(f"{fam} | {label}", w14.names(cfg)))
    path = os.path.join(args.out, "wave15_table.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    w14.RUN = w11.RUN = args.run
    w11.SEEDS = w14.SEEDS
    os.makedirs(args.out, exist_ok=True)
    fig_false9(args)
    table(args)


if __name__ == "__main__":
    main()
