"""
Final report of the snap9 experiments (Dave, 5 Nov 2026): the search over each architecture, the best model of each,
and the six metrics. All runs: 75% nines, data seed 0, 5 seeds.

Metrics (one name each; phase 1 = the stream up to the 64th nine; phase 2 = digit 7 at 75%, no nines):
  Memorisation    mean, at 1, 2, 4, 8, 16, 32, 64 nines presented, of recall on the training nines shown so far
  Generalisation  mean, at the same points, of recall on held-out nines
  Protection      worst accuracy on digits 0-8 over phase 1 (every checkpoint, including just after a sleep)
  Resilience      mean, at 1, 2, 4, 8, 16, 32, 64 sevens presented in phase 2, of recall on held-out 7s
  Retention       recall on held-out nines at the end of phase 2 (no nines presented in phase 2)
  Protection-2    worst accuracy on the digits other than 7 and 9 over phase 2

    python -W ignore -m single_stream.plot_final --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import json
import os

import numpy as np

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, save, style

RUN = None
SEEDS = range(5)
POINTS = (1, 2, 4, 8, 16, 32, 64)
P1 = 64
NAMES = ("Memorisation", "Generalisation", "Protection", "Resilience", "Retention", "Protection-2")

# the search, per family: (config, short label, kind); kind: base search, sleep (+ reset), sleep kept, replay
STM_SEARCH = [("w13_r64", "r 64", "base"), ("w13_r256_T4", "r 256 T 4", "base"), ("w13_adapt256", "adapt", "base"),
              ("w13_adapt256_prio", "adapt + prio", "base"), ("w13_adapt256_prio_first2", "first 2", "base"),
              ("w13_adapt256_prio_fstop9", "first stop .9", "base"),
              ("w13_adapt256_prio_first2_fstop9", "first 2 + stop .9", "base"),
              ("w13_adapt256_prio_stop9", "stop .9", "base"), ("w13_adapt256_prio_first32", "first 32", "base"),
              ("w13_adapt256_prio_replay255", "replay 255", "base"),
              ("w15_stm_rps64", "replay 64/sample", "base"), ("w15_stm_rps128", "replay 128/sample", "base"),
              ("w16_stm_sl16_1e-4_k25_reset", "sleep gentle", "sleep"),
              ("w16_stm_sl16_1e-4_k100_reset", "sleep middle", "sleep"),
              ("w16_stm_sl16_3e-4_k100_reset", "sleep strong", "sleep"),
              ("w16_stm_sl16_1e-4_k25_keep", "kept, gentle", "kept"),
              ("w16_stm_sl16_1e-4_k100_keep", "kept, middle", "kept"),
              ("w16_stm_sl16_3e-4_k100_keep", "kept, strong", "kept")]
LTM_SEARCH = [("w13_ltm_every1", "Adam 1e-3, every sample", "base"), ("w13_ltm_every16", "Adam 1e-3, every 16", "base"),
              ("w13_b16_e1_sgd_lr0.01", "SGD .01, every sample", "base"),
              ("w13_b16_e4_lr1e-4", "Adam 1e-4, every 4", "base"),
              ("w13_b1_e1_rp15_head_lr3e-3", "head, + replay, 3e-3", "base"),
              ("w14_ltm_o1_rp15_lr1e-4_k4", "balanced", "base"),
              ("w14_ltm_o1_rp15_lr1e-4_k1", "+ replay, 1 step", "base"),
              ("w14_ltm_o1_rp15_head_lr1e-3", "head, + replay, 1e-3", "base"),
              ("w14_ltm_o2_lr3e-5", "gentlest", "base"), ("w14_ltm_o2_lr1e-4", "current, 1e-4", "base"),
              ("w14_ltm_o2_lr3e-4", "current, 3e-4", "base"),
              ("w16_ltm_3e-5_sl16_1e-4_k25", "gentlest + sleep gentle", "sleep"),
              ("w16_ltm_3e-5_sl16_1e-4_k100", "gentlest + sleep middle", "sleep"),
              ("w16_ltm_3e-5_sl16_3e-4_k100", "gentlest + sleep strong", "sleep"),
              ("w16_ltm_k4_sl16_1e-4_k25", "balanced + sleep gentle", "sleep"),
              ("w16_ltm_k4_sl16_1e-4_k100", "balanced + sleep middle", "sleep"),
              ("w16_ltm_k4_sl16_3e-4_k100", "balanced + sleep strong", "sleep")]
HYB_SEARCH = [("w14_hyb_o2_lr3e-5", "gentlest LTM", "base"), ("w14_hyb_o2_lr1e-4", "current 1e-4", "base"),
              ("w14_hyb_o2_lr3e-4", "current 3e-4", "base"), ("w14_hyb_o1_rp15_lr1e-4_k1", "+ replay 1 step", "base"),
              ("w14_hyb_o1_rp15_lr1e-4_k4", "balanced LTM", "base"), ("w14_hyb_o1_rp15_head_lr1e-3", "head", "base"),
              ("w15_hyb_o2_lr3e-5_rps64", "gentlest, replay 64", "replay"),
              ("w15_hyb_o2_lr3e-5_rps128", "gentlest, replay 128", "replay"),
              ("w15_hyb_o2_lr3e-5_u64", "gentlest, uniform 64", "replay"),
              ("w15_hyb_o1_rp15_lr1e-4_k4_rps64", "balanced, replay 64", "replay"),
              ("w15_hyb_o1_rp15_lr1e-4_k4_rps128", "balanced, replay 128", "replay"),
              ("w15_hyb_o1_rp15_lr1e-4_k4_u64", "balanced, uniform 64", "replay")]
# the best model of each architecture: (label, phase-1 runs, phase-2 runs (experiment B), colour, line style)
BEST = [("STM only", "w16_stm_sweep64", "w16b_stm_none", INK, "-"),
        ("STM + sleep + reset", "w16_stm_sl16_1e-4_k100_reset", "w16b_stm_middle_reset", SERIES[1], "-"),
        ("LTM only, balanced", "w16_ltm_k4_sweep64", "w16b_ltm_k4_none", SERIES[3], "-"),
        ("LTM only, gentlest + sleep", "w16_ltm_3e-5_sl16_1e-4_k25", "w16b_ltm_3e-5_gentle", SERIES[2], "-"),
        ("hybrid, balanced LTM", "w15_hyb_o1_rp15_lr1e-4_k4_base", "w16b_hyb_k4_none", SERIES[4], "-"),
        ("hybrid, gentlest LTM", "w15_hyb_o2_lr3e-5_base", "w16b_hyb_3e-5_none", SERIES[4], "--")]


def hists(cfg):
    return [json.load(open(os.path.join(RUN, f"{cfg}_s{s}.json")))["result"]["history"] for s in SEEDS]


def at(h, key, n, count="new_samples", extra=None):
    """The last entry (the state going forward, so after a sleep) with count == n."""
    es = [e for e in h if e.get(count) == n and (extra is None or extra(e))]
    return np.nan if not es or es[-1].get(key) is None else es[-1][key]


def phase1(H):
    """(Memorisation, Generalisation, Protection), seed means; phase 1 = up to the 64th nine, before any phase 2."""
    p1 = lambda e: not e.get("new2_samples")
    m = np.nanmean([[at(h, "mem_seen", n, extra=p1) for n in POINTS] for h in H])
    g = np.nanmean([[at(h, "new_recall", n, extra=p1) for n in POINTS] for h in H])
    p = np.mean([min(e["others_acc"] for e in h if e.get("others_acc") is not None and e.get("new_samples", 0) <= P1
                     and p1(e)) for h in H])
    return m, g, p


def phase2(H):
    """(Resilience, Retention, Protection-2), seed means."""
    r = np.nanmean([[at(h, "new2_recall", n, count="new2_samples") for n in POINTS] for h in H])
    t = np.mean([h[-1]["new_recall"] for h in H])
    p = np.mean([min(e["others2_acc"] for e in h if e.get("new2_samples", 0) > 0) for h in H])
    return r, t, p


def curve(H, key, ns, count="new_samples", extra=None):
    return np.array([np.nanmean([at(h, key, n, count, extra) for h in H]) for n in ns])


KIND = {"base": ("o", True), "sleep": ("s", True), "kept": ("s", False), "replay": ("^", True)}
KIND_LABEL = {"base": "search (no sleep)", "sleep": "sleep at the 16th nine + reset", "kept": "sleep, STM kept",
              "replay": "more STM replay (wave 15)"}
KIND_COLOUR = {"base": SERIES[0], "sleep": SERIES[1], "kept": SERIES[1], "replay": SERIES[3]}


def fig_search(args):
    fams = [("STM only", STM_SEARCH, "w13_adapt256_prio_fstop9", "w16_stm_sl16_1e-4_k100_reset"),
            ("LTM only", LTM_SEARCH, "w14_ltm_o1_rp15_lr1e-4_k4", "w16_ltm_3e-5_sl16_1e-4_k25"),
            ("Hybrid (STM + learning LTM)", HYB_SEARCH, "w14_hyb_o1_rp15_lr1e-4_k4", "w14_hyb_o2_lr3e-5")]
    stm_ref = phase1(hists("w13_adapt256_prio_fstop9"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.4), facecolor=SURFACE, sharex=True, sharey="row")
    rows = []
    for col, (title, search, *picks) in enumerate(fams):
        seen = set()
        for cfg, label, kind in search:
            m, g, p = phase1(hists(cfg))
            rows.append((title, label, kind, m, g, p))
            marker, filled = KIND[kind]
            c = KIND_COLOUR[kind]
            for row, v in ((0, g), (1, m)):
                ax = axes[row, col]
                ax.scatter(p, v, s=70, marker=marker, zorder=3, lw=1.6, edgecolor=c if not filled else SURFACE,
                           facecolor=c if filled else SURFACE, label=KIND_LABEL[kind] if kind not in seen else None)
                if cfg in picks and row == 0:
                    ax.annotate(label, (p, v), xytext=(6, 6), textcoords="offset points",
                                fontsize=8.5, color=INK)
            seen.add(kind)
        for row, v in ((0, stm_ref[1]), (1, stm_ref[0])):
            axes[row, col].scatter(stm_ref[2], v, s=220, marker="*", color=INK, zorder=4,
                                   label="STM only (reference)" if row == 0 else None)
        style(axes[0, col], f"{title}\nGeneralisation", xlabel="")
        style(axes[1, col], "Memorisation", xlabel="Protection (worst digits 0-8, phase 1)")
        axes[0, col].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    for ax in axes.flat:
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, 1.03)
    fig.suptitle("The search: each point one configuration (mean of 5 seeds); up and right is better",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "final_search.png")
    return rows


def fig_phase1(args):
    ns = list(range(1, P1 + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    p1 = lambda e: not e.get("new2_samples")
    for label, cfg, _, col, ls in BEST:
        H = hists(cfg)
        for ax, key in zip(axes, ("mem_seen", "new_recall", "others_acc")):
            ax.plot(ns, curve(H, key, ns, extra=p1), color=col, ls=ls, lw=2.4 if "reset" in label else 1.8,
                    label=label, zorder=3)
    for ax, t in zip(axes, ("Memorisation (training nines shown so far)", "Generalisation (held-out nines)",
                            "Protection (digits 0-8)")):
        style(ax, t, xlabel="nines presented (phase 1)")
        ax.set_xscale("log")
        ax.set_xticks(POINTS)
        ax.set_xticklabels([str(n) for n in POINTS])
        ax.axvline(16, color=GREY, lw=1, ls=":", zorder=1)
    axes[0].set_ylim(0, 1.03)
    axes[1].set_ylim(0, 1.0)
    axes[2].set_ylim(0.88, 0.975)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Phase 1: the best model of each architecture (mean of 5 seeds; sleeps at the 16th nine, dotted)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args, "final_phase1.png")


def fig_phase2(args):
    """Phase 2 (experiment B): sleeps, if any, are between the phases. x: 7s presented, the end at END."""
    end = 153
    ns = list(range(0, 65))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for label, _, cfg, col, ls in BEST:
        H = hists(cfg)
        for ax, key in zip(axes, ("new2_recall", "new_recall", "others2_acc")):
            y = [np.nanmean([at(h, key, n, "new2_samples", lambda e: n > 0 or e.get("new_samples") == P1)
                             for h in H]) for n in ns]
            y.append(np.mean([h[-1][key] for h in H]))
            ax.plot(ns + [end], y, color=col, ls=ls, lw=2.4 if "reset" in label else 1.8, label=label, zorder=3)
    for ax, t in zip(axes, ("Resilience (held-out 7s)", "Retention (held-out nines; no nines in phase 2)",
                            "Protection-2 (digits other than 7 and 9)")):
        style(ax, t, xlabel="7s presented in phase 2 (0 = end of phase 1, after any sleep)")
        ax.set_xscale("symlog", linthresh=1)
        t = [0, 1, 2, 4, 8, 16, 32, 64, end]
        ax.set_xticks(t)
        ax.set_xticklabels([str(v) for v in t[:-1]] + ["end"])
    axes[0].set_ylim(0.78, 1.0)
    axes[1].set_ylim(0.5, 1.0)
    axes[2].set_ylim(0.9, 0.975)
    axes[1].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Phase 2: digit 7 at 75%, no nines; the same models (mean of 5 seeds)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args, "final_phase2.png")


SLEEP_AT = 16


def at_sleep(H, n=SLEEP_AT):
    """(Memorisation, Generalisation, digits 0-8) right before and right after the sleep at the n-th nine: two
    3-vectors, seed means (point values, not the phase summaries)."""
    keys = ("mem_seen", "new_recall", "others_acc")
    pick = lambda h, after: [e for e in h if e.get("new_samples") == n and bool(e.get("after_sleep")) == after][-1]
    return [np.mean([[pick(h, after)[k] for k in keys] for h in H], 0) for after in (False, True)]


def summary(args):
    """Rows of the summary: each best model (six metrics), and under each model with a sleep in phase 1 a row with
    its values at the 16th nine just before and just after the sleep (first three metrics only)."""
    S = []
    for label, c1, c2, col, ls in BEST:
        H1 = hists(c1)
        S.append((label, col, ls, phase1(H1) + phase2(hists(c2)), None))
        if any(e.get("after_sleep") for e in H1[0]):
            pre, post = at_sleep(H1)
            S.append((f"  {SLEEP_AT} nines: pre \u2192 post sleep", col, ls, None, (pre, post)))
    fig, axes = plt.subplots(1, 6, figsize=(16, 4.6), facecolor=SURFACE, sharey=True)
    ys = np.arange(len(S))[::-1]
    fmt = lambda j, x: f"{x:.2f}" if j not in (2, 5) else f"{x:.3f}"
    for j, ax in enumerate(axes):
        vals = []
        for y, (label, col, ls, v, pp) in zip(ys, S):
            if v is not None:
                vals.append(v[j])
                ax.plot([0, v[j]], [y, y], color=GREY, lw=0.8, zorder=1)
                ax.scatter(v[j], y, s=80, color=col, marker="o" if ls == "-" else "D", zorder=3, edgecolor=SURFACE)
                ax.annotate(fmt(j, v[j]), (v[j], y), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=8, color=INK)
            elif j < 3:
                a, b = pp[0][j], pp[1][j]
                vals += [a, b]
                ax.annotate("", (b, y), (a, y), arrowprops=dict(arrowstyle="->", color=col, lw=1.2), zorder=2)
                ax.scatter(a, y, s=60, facecolor=SURFACE, edgecolor=col, lw=1.5, zorder=3)
                ax.scatter(b, y, s=60, color=col, edgecolor=SURFACE, zorder=3)
                if abs(b - a) < 1e-3:
                    ax.annotate(fmt(j, b), (b, y), xytext=(0, 7), textcoords="offset points", ha="center",
                                fontsize=7.5, color=INK)
                else:
                    for x, dx in ((a, -4 if b > a else 4), (b, 4 if b > a else -4)):
                        ax.annotate(fmt(j, x), (x, y), xytext=(dx, 6), textcoords="offset points",
                                    ha="right" if dx < 0 else "left", fontsize=7.5, color=INK)
        lo, hi = min(vals), max(vals)
        span = hi - lo
        ax.set_xlim(lo - 0.25 * span - 0.01, hi + 0.25 * span + 0.01)
        style(ax, NAMES[j], xlabel="")
        ax.set_ylim(-0.6, len(S) - 0.3)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([s[0] for s in S])
    for t, (_, _, _, v, _) in zip(axes[0].get_yticklabels(), S):
        if v is None:
            t.set_fontsize(8.5)
            t.set_color(GREY)
    fig.suptitle("The six metrics for the best model of each architecture (mean of 5 seeds; each axis zoomed to its "
                 "own range); open = just before the sleep, filled = just after", color=INK, x=0.01, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, args, "final_summary.png")
    return S


def table(args, rows, S):
    L = ["## The search (phase 1, 75% nines, mean of 5 seeds)", "",
         "| family | configuration | kind | Memorisation | Generalisation | Protection |", "|---|---|---|---|---|---|"]
    for fam, label, kind, m, g, p in rows:
        L.append(f"| {fam} | {label} | {kind} | {m:.2f} | {g:.2f} | {p:.3f} |")
    L += ["", "## Best model of each architecture", "", "| model | " + " | ".join(NAMES) + " |",
          "|---|" + "---|" * len(NAMES)]
    for label, _, _, v, pp in S:
        if v is not None:
            L.append(f"| {label} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
        else:
            L.append(f"| {label.strip()} | " + " | ".join(f"{a:.3f} -> {b:.3f}" for a, b in zip(*pp))
                     + " | - | - | - |")
    path = os.path.join(args.out, "final_tables.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(path)


RECAP = [("ltm", "LTM only (no buffer)", INK, ":"), ("ltm_cbp", "LTM + continual backprop", GREY, ":"),
         ("fifo", "LTM + replay, FIFO buffer", SERIES[3], "--"),
         ("pri_ae3_ar1_g1", "LTM + replay, priority buffer + prioritised replay", SERIES[0], "-")]


def fig_recap(args, tables):
    """Task 1 round 4 stage C (Dohare MLP 3x2000 on MNIST, buffer capacity 500, 3 seeds), read from its tables:
    anytime accuracy against cluster size, and the few-shot balanced score at c = 128."""
    few, final = {}, {}
    for line in open(tables):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 12 and c[0].isdigit() and c[1] in ("500", "-") and line.count("|") == 13:
            few.setdefault((int(c[0]), c[2]), [float(x) for x in c[4:]])
        elif len(c) == 6 and c[0].isdigit() and c[1] in ("500", "-"):
            final[(int(c[0]), c[2])] = float(c[5].split()[0])
    ns = (1, 8, 16, 32, 64, 128, 256, 512)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    for cfg, label, col, ls in RECAP:
        axes[0].plot((32, 128, 512), [final[(c, cfg)] for c in (32, 128, 512)], color=col, ls=ls, lw=2, marker="o",
                     ms=5, label=label)
        axes[1].plot(ns, few[(128, cfg)], color=col, ls=ls, lw=2, marker="o", ms=5, label=label)
    style(axes[0], "Anytime accuracy (all digits, over the stream)", xlabel="cluster size c")
    axes[0].set_xscale("log")
    axes[0].set_xticks((32, 128, 512))
    axes[0].set_xticklabels(["32", "128", "512"])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(axes[1], "Few-shot: balanced score of a digit at its n-th exposure, c = 128 (chance .5)",
          xlabel="exposures of the digit so far (n)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ns)
    axes[1].set_xticklabels([str(n) for n in ns])
    axes[1].axhline(0.5, color=GREY, lw=1)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Before the STM: replay rescues the LTM on a clustered stream, but not in the few-shot regime "
                 "(Dohare MLP, MNIST, buffer 500, mean of 3 seeds)", color=INK, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args, "final_recap.png")


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recap", default="", help="task 1 round 4 stage C tables (stage_c_tables.md)")
    args = ap.parse_args()
    RUN = args.run
    os.makedirs(args.out, exist_ok=True)
    rows = fig_search(args)
    fig_phase1(args)
    fig_phase2(args)
    S = summary(args)
    table(args, rows, S)
    if args.recap:
        fig_recap(args, args.recap)


if __name__ == "__main__":
    main()
