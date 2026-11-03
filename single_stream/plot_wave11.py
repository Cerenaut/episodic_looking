"""
Waves 11 and 12 (Dave, 3 Nov 2026 evening): every model measured from one shot, on the same 200-sample streams.
Three qualities, against nines presented: memorised (recall on the training nines shown so far), few-shot
generalisation (held-out nines), protection (digits 0-8). Values are read at the checkpoint right after the n-th nine
presentation (--nine-checkpoints; no interpolation).

  wave11_stm_all.png       all STM configs (5 seeds): references and diagnostics; the wave-10 adaptive variants
  wave11_ltm_tradeoff.png  every LTM-only config (wave 12, 3 seeds) as a point: few-shot and memorised vs protection,
                           with the STM configs on the same seeds
  wave11_best_vs_ltm.png   the best STM configs against a selection of LTM tradeoff points, seeds 0-2 (paired)
  wave11_episodes.png      STM recall against thinking time (learning episodes)
  wave11_table.md          STM table, paired diagnostics, LTM table with the pre-registered dominance test

  python -m single_stream.plot_wave11 --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import glob
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, at, hist, save, style

KEYS = ("mem_seen", "new_recall", "others_acc")
TITLES = ("Memorised: recall on the training nines shown so far", "Generalised: recall on held-out nines",
          "Protected: accuracy on digits 0-8")
NINES = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24, 32, 40)
TICKS = (1, 2, 5, 10, 20, 40)
# pre-registered (ledger, wave 12): memorised >= at 1, 2, 5 nines; held-out >= at 1, 2, 5, 9; digits 0-8 >= at every
# checkpoint within -0.005
MEM_AT, HELD_AT, TOL = (1, 2, 5), (1, 2, 5, 9), 0.005

STM = {"r 64": "r64", "r 256 T 4": "r256_T4", "adapt 256": "adapt256", "adapt 256 + prio": "adapt256_prio",
       "+ prio, first 2 blocks": "adapt256_prio_first2", "+ prio, first stop .9": "adapt256_prio_fstop9",
       "+ prio, first 2 + first stop .9": "adapt256_prio_first2_fstop9", "+ prio, stop .9": "adapt256_prio_stop9",
       "+ prio, first 32 blocks (diag)": "adapt256_prio_first32",
       "+ prio, replay 255 per sample (diag)": "adapt256_prio_replay255"}
# figure rows: (label, color, linestyle, marker)
ROW_REF = [("r 64", INK, "--", "s"), ("r 256 T 4", SERIES[2], "--", "s"), ("adapt 256", GREY, "-", "o"),
           ("+ prio, first 32 blocks (diag)", SERIES[3], "-.", "^"),
           ("+ prio, replay 255 per sample (diag)", SERIES[1], "-.", "^")]
ROW_W10 = [("adapt 256", GREY, "-", "o"), ("adapt 256 + prio", SERIES[0], "-", "o"),
           ("+ prio, first 2 blocks", SERIES[3], "-", "o"), ("+ prio, first stop .9", SERIES[2], "-", "o"),
           ("+ prio, first 2 + first stop .9", SERIES[4], "-", "o"), ("+ prio, stop .9", INK, ":", "o")]
# chosen from the table (see main): best few-shot at any cost, and best at adaptive cost
BEST = [("STM + prio, replay 255 per sample", "adapt256_prio_replay255", SERIES[1]),
        ("STM + prio, first stop .9", "adapt256_prio_fstop9", SERIES[0])]
# LTM tradeoff points for the comparison figure, chosen from the tradeoff plot: (label, config, color, linestyle)
PICKS = [("LTM every sample: Adam 1e-3 (wave-8 baseline)", "w11_ltm_every1", INK, ":"),
         ("LTM fast: SGD .01, window 16, every sample", "w12_b16_e1_sgd_lr0.01", SERIES[2], "--"),
         ("LTM balanced: replay 15, 4 Adam steps 1e-4", "w12_b1_e1_rp15_lr1e-4_k4", SERIES[3], "--"),
         ("LTM head only: replay 15, Adam 3e-3", "w12_b1_e1_rp15_head_lr3e-3", SERIES[4], "--"),
         ("LTM protective: window 16 every 4, Adam 1e-4", "w12_b16_e4_lr1e-4", SERIES[5], "--")]
XMIN = 0.78  # tradeoff plot: protection axis starts here


def names(cfg, seeds):
    return [n for n in (f"{cfg}_s{s}" for s in seeds) if os.path.exists(os.path.join(RUN, f"{n}.json"))]


def at_nines(h, k, nines=NINES):
    """Key k at the checkpoint right after the n-th nine presentation (NaN if undefined or not reached)."""
    out = []
    for n in nines:
        e = next((e for e in h if e["new_samples"] == n), None)
        out.append(np.nan if e is None or e.get(k) is None else e[k])
    return np.array(out, dtype=float)


def mean_curves(ns):
    """Seed mean, (len(NINES), 3) for KEYS."""
    return np.nanmean([np.stack([at_nines(hist(RUN, n), k) for k in KEYS], 1) for n in ns], 0)


def summary(ns):
    """Seed means of the three qualities: memorised at MEM_AT, held-out at HELD_AT, worst digits 0-8 (1-40 nines)."""
    c = mean_curves(ns)
    i = lambda xs: [NINES.index(x) for x in xs]
    return {"mem": c[i(MEM_AT), 0].mean(), "held": c[i(HELD_AT), 1].mean(), "prot": np.nanmin(c[:, 2]),
            "held40": c[-1, 1], "curves": c}


def dominance(ltm_cfg, stm_cfg, seeds=range(3)):
    """Pre-registered test on seed means over the seeds both configs have (paired streams); returns (passes, worst
    margins mem / held / protection, seeds)."""
    s = [x for x in seeds if names(ltm_cfg, [x]) and names(stm_cfg, [x])]
    ltm, stm = names(ltm_cfg, s), names(stm_cfg, s)
    a, b = mean_curves(ltm), mean_curves(stm)
    i = lambda xs: [NINES.index(x) for x in xs]
    dm = (a[i(MEM_AT), 0] - b[i(MEM_AT), 0]).min()
    dh = (a[i(HELD_AT), 1] - b[i(HELD_AT), 1]).min()
    # every checkpoint: a seed's LTM and STM runs share the checkpoint schedule (same stream), seeds do not; seed
    # mean of the paired difference at every checkpoint of any seed, each run held at its last checkpoint <= s
    hl, hs = [hist(RUN, n) for n in ltm], [hist(RUN, n) for n in stm]
    xs = sorted({e["samples"] for h in hl + hs for e in h})
    dp = min(np.mean([at(a, x, "others_acc") - at(b, x, "others_acc") for a, b in zip(hl, hs)]) for x in xs)
    return dm >= 0 and dh >= 0 and dp >= -TOL, dm, dh, dp, len(s)


def ltm_configs():
    cfgs = sorted({os.path.basename(f)[:-len("_s0.json")] for f in glob.glob(os.path.join(RUN, "w12_*_s0.json"))})
    return ["w11_ltm_every1", "w11_ltm_every16"] + cfgs


def family(cfg):
    """(family label, color, marker) for the tradeoff plot."""
    if "head" in cfg:
        return "head only (Adam)", SERIES[5], "D"
    if "sgd" in cfg:
        return "SGD momentum .5", SERIES[3], "v"
    if "rp15" in cfg:
        return "seen replay 15 per update", SERIES[1], "^"
    if "_k" in cfg:
        return "4 or 16 steps per update", SERIES[4], "P"
    if cfg.startswith("w11_"):
        return "wave 11 baselines (every 1, every 16)", INK, "X"
    return "window / cadence x lr (Adam)", SERIES[0], "o"


def axes_nines(axes):
    for ax, t in zip(axes, TITLES):
        style(ax, t, xlabel="nines presented")
        ax.set_xscale("log")
        ax.set_xticks(TICKS, [str(n) for n in TICKS])
        ax.set_xlim(0.9, 44)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)


def plot_curves(axes, label, c, color, ls, marker, lw=2):
    for j, ax in enumerate(axes):
        ax.plot(NINES, c[:, j], color=color, lw=lw, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1, label=label,
                zorder=3)


def fig_stm_all(args):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=SURFACE)
    for row, groups, title in ((0, ROW_REF, "references and diagnostics"), (1, ROW_W10, "adaptive variants")):
        for label, color, ls, marker in groups:
            ns = names(f"w11_{STM[label]}", range(5))
            plot_curves(axes[row], f"{label} ({len(ns)} seeds)", mean_curves(ns), color, ls, marker)
        axes_nines(axes[row])
        axes[row, 2].set_ylim(0.93, 0.97)
        axes[row, 2].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK, title=title,
                            title_fontsize=9)
    fig.suptitle("Wave 11: all STM configs, mean of 5 seeds (stream = seed), read right after each nine presentation",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "wave11_stm_all.png")


def fig_tradeoff(args, rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)
    seen = set()
    off = [r for r in rows if r["prot"] < XMIN]  # collapsed regimes: listed, not plotted
    for r in rows:
        if r["prot"] < XMIN:
            continue
        fam, color, marker = family(r["cfg"])
        for ax, k in zip(axes, ("held", "mem")):
            ax.scatter(r["prot"], r[k], s=42, color=color, marker=marker, edgecolor=SURFACE, lw=0.8, zorder=3,
                       label=None if fam in seen else fam)
        seen.add(fam)
    for label, cfg in STM.items():
        s = summary(names(f"w11_{cfg}", range(3)))
        best = any(cfg == b[1] for b in BEST)
        for ax, k in zip(axes, ("held", "mem")):
            ax.scatter(s["prot"], s[k], s=150 if best else 60, color=INK if best else "none", marker="*",
                       edgecolor=INK, lw=1, zorder=4, label=None if ax is axes[1] or label != "r 64" else
                       "STM configs (filled: the two chosen)")
    for label, cfg, color, ls in PICKS:
        r = next(r for r in rows if r["cfg"] == cfg)
        tag = label.split(":")[0].split(",")[0].replace("LTM ", "")
        for ax, k in zip(axes, ("held", "mem")):
            ax.annotate(tag, (r["prot"], r[k]), xytext=(5, -3), textcoords="offset points", fontsize=8, color=INK)
    for ax, t in zip(axes, ("Few-shot: held-out recall, mean at 1, 2, 5, 9 nines",
                            "Memorised: shown-so-far recall, mean at 1, 2, 5 nines")):
        style(ax, t, xlabel="protection: worst accuracy on digits 0-8, 1 to 40 nines")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(XMIN, 0.975)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    if off:
        fig.text(0.01, 0.01, f"Not shown, digits 0-8 below {XMIN}: " + ", ".join(r["cfg"][4:] for r in off)
                 + ". Labelled: LTM regimes in the comparison figure.",
                 fontsize=8, color=INK)
    fig.suptitle("LTM-only regimes (wave 12) and STM configs: seeds 0-2 on the same streams; up and right is better",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save(fig, args, "wave11_ltm_tradeoff.png")


def fig_best_vs_ltm(args):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURFACE)
    for label, cfg, color, ls in PICKS:
        plot_curves(axes, label, mean_curves(names(cfg, range(3))), color, ls, None, lw=1.6)
    for label, cfg, color in BEST:
        plot_curves(axes, label, mean_curves(names(f"w11_{cfg}", range(3))), color, "-", "o", lw=2.4)
    axes_nines(axes)
    axes[2].set_ylim(0.84, 0.975)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.suptitle("Best STM configs vs LTM-only tradeoff points, mean of seeds 0-2 (paired streams)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    save(fig, args, "wave11_best_vs_ltm.png")


def fig_episodes(args):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for label, color, ls, marker in ROW_REF + ROW_W10[1:]:
        ns = names(f"w11_{STM[label]}", range(5))
        eps = np.mean([at_nines(hist(RUN, n), "learning_episodes") for n in ns], 0)
        c = mean_curves(ns)
        for j, ax in enumerate(axes):
            ax.plot(eps, c[:, j], color=color, lw=1.8, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1, label=label,
                    zorder=3)
    for ax, t in zip(axes, TITLES[:2]):
        style(ax, t, xlabel="learning episodes (thinking time), mean over seeds")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
    axes[1].legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK)
    fig.suptitle("Wave 11: recall against thinking time; markers at 1, 2, 3, ... 40 nines", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args, "wave11_episodes.png")


def paired(a, b, k, nines, seeds):
    """Mean and SE over seeds of a - b (configs) for key k at nines; seeds both configs have."""
    s = [x for x in seeds if os.path.exists(os.path.join(RUN, f"w11_{a}_s{x}.json"))
         and os.path.exists(os.path.join(RUN, f"w11_{b}_s{x}.json"))]
    d = np.array([at_nines(hist(RUN, f"w11_{a}_s{x}"), k, nines) - at_nines(hist(RUN, f"w11_{b}_s{x}"), k, nines)
                  for x in s])
    return d.mean(0), d.std(0, ddof=1) / np.sqrt(len(s)), len(s)


def table(args, rows):
    f = lambda v: " / ".join(f"{x:.2f}" for x in v)
    G = (1, 2, 3, 5, 9, 16, 40)
    i = [NINES.index(x) for x in G]
    L = ["## STM configs (wave 11, mean of 5 seeds; values right after the n-th nine)", "",
         "| config | seeds | memorised 1/2/3/5/9 | held-out 1/2/3/5/9/16/40 | digits 0-8 worst / at 40 | "
         "learning episodes @200 samples |", "|---|---|---|---|---|---|"]
    for label, cfg in STM.items():
        ns = names(f"w11_{cfg}", range(5))
        c = mean_curves(ns)
        eps = np.mean([hist(RUN, n)[-1]["learning_episodes"] for n in ns])
        L.append(f"| {label} | {len(ns)} | {f(c[i[:5], 0])} | {f(c[i, 1])} | {np.nanmin(c[:, 2]):.3f} / "
                 f"{c[-1, 2]:.3f} | {eps:,.0f} |")
    L += ["", "## Paired differences in held-out recall (mean +- SE over seeds) at 1 / 2 / 3 / 5 / 9 nines", "",
          "| comparison | seeds | difference |", "|---|---|---|"]
    for a, b in (("r256_T4", "adapt256_prio"), ("adapt256_prio_first32", "adapt256_prio"),
                 ("adapt256_prio_replay255", "adapt256_prio"), ("adapt256_prio_replay255", "r256_T4"),
                 ("adapt256_prio", "adapt256"), ("adapt256_prio_fstop9", "adapt256_prio")):
        m, se, n = paired(a, b, "new_recall", (1, 2, 3, 5, 9), range(5))
        L.append(f"| {a} - {b} | {n} | " + " / ".join(f"{x:+.2f}+-{e:.2f}" for x, e in zip(m, se)) + " |")
    L += ["", "## LTM-only configs (wave 12 and the wave-11 baselines), mean of seeds 0-2, and the pre-registered test",
          "",
          "Test: beats a STM config if memorised >= at 1, 2, 5 nines, held-out >= at 1, 2, 5, 9 nines and digits 0-8 "
          ">= (within -0.005) at every checkpoint, on seed means of the same seeds 0-2. Margins are the worst "
          "LTM - STM difference for each quality.", "",
          "| config | memorised 1/2/5 | held-out 1/2/5/9/40 | digits 0-8 worst | "
          + " | ".join(f"vs {b[0]}: pass (mem / held / digits 0-8 margins)" for b in BEST) + " |",
          "|---|---|---|---|" + "---|" * len(BEST)]
    j = [NINES.index(x) for x in (1, 2, 5, 9, 40)]
    for r in rows:
        c = r["curves"]
        cells = []
        for _, cfg, _ in BEST:
            ok, dm, dh, dp, n = r["dom"][cfg]
            cells.append(f"{'**yes**' if ok else 'no'} ({dm:+.2f} / {dh:+.2f} / {dp:+.3f})" + ("" if n == 3 else
                                                                                         f" [{n} seeds]"))
        L.append(f"| {r['cfg']} | {f(c[[0, 1, 4], 0])} | {f(c[j, 1])} | {r['prot']:.3f} | " + " | ".join(cells)
                 + " |")
    path = os.path.join(args.out, "wave11_table.md")
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
    rows = []
    for cfg in ltm_configs():
        ns = names(cfg, range(3))
        r = {"cfg": cfg, **summary(ns), "dom": {}}
        for _, b, _ in BEST:
            r["dom"][b] = dominance(cfg, f"w11_{b}")
        rows.append(r)
    fig_stm_all(args)
    fig_tradeoff(args, rows)
    fig_best_vs_ltm(args)
    fig_episodes(args)
    table(args, rows)


RUN = None

if __name__ == "__main__":
    main()
