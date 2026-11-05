"""
Wave 16a: consolidation (sleep) within phase 1 (75% nines, stream cut after the 64th nine). Stage 1 sleeps once at the
end (64th nine, --sleep-sweep); stage 2 sleeps at the 16th nine, runs on to the 64th and sleeps again with the same
setting (sweep reading). STM only with the STM kept or reset after each sleep; LTM only, gentlest (Adam 3e-5) and
balanced (replay 15, 4 steps Adam 1e-4).

    python -W ignore -m single_stream.plot_wave16 --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, hist, save, style

RUN = None
SEEDS = range(5)
NINES = list(range(1, 65))
TICKS = (1, 2, 5, 10, 16, 20, 40, 64)
KEYS = ("mem_seen", "new_recall", "others_acc", "others_as_new")
TITLES = ("memorised (shown-so-far nines)", "held-out nines (few-shot recall)", "digits 0-8 (protection)",
          "digits 0-8 answered \"9\"")
# the three stage-2 settings (lr, steps), in fixed colour order
SETTINGS = [("1e-4_k25", 1e-4, 25, "gentle: Adam 1e-4, 25 steps"), ("1e-4_k100", 1e-4, 100, "middle: Adam 1e-4, 100 steps"),
            ("3e-4_k100", 3e-4, 100, "strong: Adam 3e-4, 100 steps")]
LTMS = [("3e-5", "LTM only, gentlest (current input, Adam 3e-5)"),
        ("k4", "LTM only, balanced (replay 15, 4 steps Adam 1e-4)")]
SLEEP_AT = 16


def runs(cfg):
    return [hist(RUN, f"{cfg}_s{s}") for s in SEEDS]


def results(cfg):
    import json
    return [json.load(open(os.path.join(RUN, f"{cfg}_s{s}.json")))["result"] for s in SEEDS]


def curve(H):
    """Seed mean over nines 1..64 (pre-sleep values), with the post-sleep reading inserted after the 16th nine when
    present: x (len n), y (n, len(KEYS))."""
    xs, ys = [], []
    for n in NINES:
        pre = [[e for e in h if e.get("new_samples") == n and not e.get("after_sleep")][-1] for h in H]
        xs.append(n)
        ys.append([np.mean([e[k] for e in pre]) for k in KEYS])
        post = [[e for e in h if e.get("new_samples") == n and e.get("after_sleep")] for h in H]
        if all(post):
            xs.append(n)
            ys.append([np.mean([p[-1][k] for p in post]) for k in KEYS])
    return np.array(xs), np.array(ys)


def worst(H):
    return np.mean([min(e["others_acc"] for e in h if e.get("others_acc") is not None) for h in H])


def end_sleep(cfg, lr, steps):
    """The --sleep-sweep reading (sleep at the end, from the 64th nine): seed mean of KEYS, STM reset if read, else
    kept (LTM only has no STM)."""
    out = []
    for r in results(cfg):
        sw = [s for s in r["sleep_sweep"] if np.isclose(s["setting"]["sleep_lr"], lr)
              and s["setting"]["sleep_steps"] == steps and s["setting"]["sleep_priority"] > 0][0]
        out.append([sw["reset" if "reset" in sw else "keep"][k] for k in KEYS])
    return np.mean(out, 0)


def axes_fmt(ax, j, ylims):
    style(ax, TITLES[j], xlabel="nines presented")
    ax.set_xscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.set_ylim(*ylims[j])
    ax.axvline(SLEEP_AT, color=GREY, lw=1, ls=":", zorder=1)


def line(axes, cols, x, y, color, ls, label, lw=2):
    for ax, j in zip(axes, cols):
        ax.plot(x, y[:, j], color=color, lw=lw, ls=ls, label=label, zorder=3)


YL = [(0, 1.02), (0, 1.02), (0.74, 0.98), (0, 0.2)]


def fig_stm_reset(args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    x, y = curve(runs("w16_stm_sweep64"))
    line(axes, range(3), x, y, INK, "-", "no sleep", lw=2.4)
    for i, (k, _, _, label) in enumerate(SETTINGS):
        x, y = curve(runs(f"w16_stm_sl16_{k}_reset"))
        line(axes, range(3), x, y, SERIES[i], "-", f"sleep at 16, {label}")
    for j, ax in enumerate(axes):
        axes_fmt(ax, j, [(0.9, 1.005), (0.6, 1.0), (0.87, 0.975)])
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("STM only: one sleep at the 16th nine, then the STM is reset (mean of 5 seeds, 75% nines)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave16_stm_reset.png")


def fig_stm_keep(args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    x, y = curve(runs("w16_stm_sweep64"))
    line(axes, (1, 2, 3), x, y, INK, "-", "no sleep", lw=2.4)
    for i, (k, _, _, label) in enumerate(SETTINGS):
        for m, ls in (("reset", "-"), ("keep", "--")):
            x, y = curve(runs(f"w16_stm_sl16_{k}_{m}"))
            line(axes, (1, 2, 3), x, y, SERIES[i], ls, f"{label}, STM {'reset' if m == 'reset' else 'kept'}")
    for ax, j in zip(axes, (1, 2, 3)):
        axes_fmt(ax, j, YL)
    axes[1].set_ylim(0.74, 0.98)
    axes[0].set_ylim(0.6, 1.0)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("STM only after the sleep at 16: STM kept (dashed) vs reset (solid); keeping it double-counts "
                 "(mean of 5 seeds)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave16_stm_keep.png")


def fig_ltm(args):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor=SURFACE)
    xs, ys = curve(runs("w16_stm_sweep64"))
    for row, (L, title) in zip(axes, LTMS):
        line(row, range(3), xs, ys, GREY, "--", "STM only, no sleep", lw=1.6)
        x, y = curve(runs(f"w16_ltm_{L}_sweep64"))
        line(row, range(3), x, y, INK, "-", "no sleep", lw=2.4)
        for i, (k, _, _, label) in enumerate(SETTINGS):
            x, y = curve(runs(f"w16_ltm_{L}_sl16_{k}"))
            line(row, range(3), x, y, SERIES[i], "-", f"sleep at 16, {label}")
        for j, ax in enumerate(row):
            axes_fmt(ax, j, [(0, 1.02), (0, 1.0), (0.87, 0.975)])
        row[0].set_title(f"{title}\n{TITLES[0]}", color=INK, loc="left", fontsize=10.5)
    axes[1, 0].legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("LTM only: one sleep at the 16th nine (mean of 5 seeds, 75% nines)", color=INK, x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "wave16_ltm.png")


def points():
    """(model, schedule, setting index or None, held-out at the end, worst digits 0-8 over the whole phase incl. the
    reading after the end sleep, memorised at the end, digits 0-8 at the end)."""
    P = []
    for model, base, sl16 in [("STM only", "w16_stm_sweep64", "w16_stm_sl16_{}_reset"),
                              ("LTM gentlest", "w16_ltm_3e-5_sweep64", "w16_ltm_3e-5_sl16_{}"),
                              ("LTM balanced", "w16_ltm_k4_sweep64", "w16_ltm_k4_sl16_{}")]:
        H0 = H = runs(base)
        _, y = curve(H)
        P.append((model, "none", None, y[-1, 1], worst(H), y[-1, 0], y[-1, 2]))
        for i, (k, lr, st, _) in enumerate(SETTINGS):
            e = end_sleep(base, lr, st)
            P.append((model, "at 64", i, e[1], min(worst(H0), e[2]), e[0], e[2]))
            H = runs(sl16.format(k))
            _, y = curve(H)
            P.append((model, "at 16", i, y[-1, 1], worst(H), y[-1, 0], y[-1, 2]))
            e = end_sleep(sl16.format(k), lr, st)
            P.append((model, "at 16 and 64", i, e[1], min(worst(H), e[2]), e[0], e[2]))
    return P


MARK = {"none": "*", "at 64": "o", "at 16": "s", "at 16 and 64": "D"}


def fig_tradeoff(args, P):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURFACE, sharey=True)
    for ax, model in zip(axes, ("STM only", "LTM gentlest", "LTM balanced")):
        for m, sched, i, held, w, _, _ in P:
            if m != model:
                continue
            ax.scatter(held, w, marker=MARK[sched], s=150 if sched == "none" else 60, zorder=3, edgecolors=SURFACE,
                       linewidths=1.5, color=INK if i is None else SERIES[i])
        style(ax, f"{model}" + (" (STM reset after each sleep)" if model == "STM only" else ""),
              xlabel="held-out nines at the 64th nine (after the end sleep, if any)")
        ax.set_xlim(0.5, 1.0)
    axes[0].set_ylabel("digits 0-8, worst over the phase")
    h = [plt.Line2D([], [], ls="", marker=MARK[s], color=INK, ms=8 if s != "none" else 12, label=f"sleep {s}"
                    if s != "none" else "no sleep") for s in MARK]
    h += [plt.Line2D([], [], ls="", marker="o", color=SERIES[i], ms=8, label=label) for i, (_, _, _, label)
          in enumerate(SETTINGS)]
    axes[0].legend(handles=h, loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Sleep schedules: few-shot recall vs protection (mean of 5 seeds; up and right is better)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave16_tradeoff.png")


def table(args, P):
    L = ["## Wave 16a: sleep schedules within phase 1 (75% nines, 64 nines, mean of 5 seeds; STM reset after sleep)", "",
         "(values at 64 are after the end sleep, if any; worst is over the whole phase, including that reading)", "",
         "| model | sleep | setting | memorised at 64 | held-out at 64 | digits 0-8 at 64 | digits 0-8 worst |", "|---|---|---|---|---|---|---|"]
    for m, sched, i, held, w, mem, end in P:
        L.append(f"| {m} | {sched} | {'-' if i is None else SETTINGS[i][3]} | {mem:.2f} | {held:.2f} | {end:.3f} | {w:.3f} |")
    L += ["", "## STM only, sleep at 16: STM kept vs reset", "",
          "| setting | STM | just after sleep (mem / held / 0-8 / false 9) | at 64 | worst 0-8 | max false 9 |",
          "|---|---|---|---|---|---|"]
    for k, _, _, label in SETTINGS:
        for m in ("keep", "reset"):
            H = runs(f"w16_stm_sl16_{k}_{m}")
            x, y = curve(H)
            a = y[list(x).index(SLEEP_AT) + 1]
            f9 = np.mean([max(e["others_as_new"] for e in h if e.get("others_as_new") is not None) for h in H])
            L.append(f"| {label} | {m} | {a[0]:.2f} / {a[1]:.2f} / {a[2]:.3f} / {a[3]:.3f} | "
                     f"{y[-1, 0]:.2f} / {y[-1, 1]:.2f} / {y[-1, 2]:.3f} | {worst(H):.3f} | {f9:.3f} |")
    path = os.path.join(args.out, "wave16_table.md")
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
    fig_stm_reset(args)
    fig_stm_keep(args)
    fig_ltm(args)
    P = points()
    fig_tradeoff(args, P)
    table(args, P)


if __name__ == "__main__":
    main()
