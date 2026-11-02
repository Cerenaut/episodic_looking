"""
Wave 8 figures (what bias only; few-shot nines without replacement, seen-sample replay, --data-seed 0):

  wave8_ltm_vs_stm.png   LTM alone stepping every sample vs every 16, vs the default STM, 25% and 50% nines
  wave8_ltm_long.png     the two LTM cadences over 3000 samples: does the every-sample LTM degenerate?
  wave8_seed_grid.png    STM seed x stream seed (3 x 3): which seed drives the band
  wave8_repeats.png      repeats {16, 64, 128, 256} x episode length T {4, 8}
  wave8_rewards.png      rewards for stuck images: held-out recall, and per-nine p(9) at the last presentation
  wave8_qkv.png          offline Q/K/V search: success vs number of bias elements, per family
  wave8_table.md         numbers behind the figures

  python -m single_stream.plot_wave8 --run runs_ss/stm_frozen_20261101/snap9 \
      --qkv runs_ss/stm_frozen_20261101/qkv_search/qkv_search.json --out <dir>
"""
import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import GRID, INK, INK2, SERIES, SURFACE

KEYS = ("new_train_recall", "new_recall", "others_acc")
TITLES = ("Memorised: recall on the 16 training nines", "Generalised: recall on held-out nines",
          "Accuracy on digits 0-8")
GREY = "#8f8d87"


def hist(run, name):
    return json.load(open(os.path.join(run, f"{name}.json")))["result"]["history"]


def at(h, s, k):
    """Value of key k at the last checkpoint <= s samples."""
    c = [e for e in h if e["samples"] <= s]
    return c[-1][k] if c else np.nan


def band(run, names, xmax=None):
    """Seed mean, min and max of KEYS on the first run's checkpoints (runs share the checkpoint schedule)."""
    hs = [hist(run, n) for n in names]
    xs = [e["samples"] for e in hs[0] if xmax is None or e["samples"] <= xmax]
    v = np.array([[[at(h, x, k) if at(h, x, k) is not None else np.nan for k in KEYS] for x in xs] for h in hs])
    return np.maximum(xs, 2), v.mean(0), v.min(0), v.max(0)


def style(ax, title, xlabel="samples presented"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel(xlabel)


def curves(ax_row, run, groups, xmax=None, log=True):
    """groups: (label, names, color, linestyle, marker)."""
    for label, names, color, ls, marker in groups:
        xs, mean, lo, hi = band(run, names, xmax)
        for j, ax in enumerate(ax_row):
            ax.plot(xs, mean[:, j], color=color, lw=2, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1,
                    label=label + (f" ({len(names)} runs)" if len(names) > 1 else ""), zorder=3)
            if len(names) > 1:
                ax.fill_between(xs, lo[:, j], hi[:, j], color=color, alpha=0.12, lw=0, zorder=2)
    for ax, t in zip(ax_row, TITLES):
        style(ax, t)
        if log:
            ax.set_xscale("log")


def save(fig, args, name):
    path = os.path.join(args.out, name)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def ltm_names(e, n=5):
    return [f"w8_ltm_every{e}_ss{s}" for s in range(n)]


GRID9 = [f"w8_grid_seed{a}_ss{b}" for a in range(3) for b in range(3)]


def fig_ltm_vs_stm(args):
    rows = [("25% nines (stream without replacement)", [
                ("LTM alone, step every sample", ltm_names(1), SERIES[1], "-", "o"),
                ("LTM alone, step every 16 samples", ltm_names(16), INK, "-", "o"),
                ("STM (what, r 64, T 4), LTM frozen", GRID9, SERIES[0], "--", "s")]),
            ("50% nines (with replacement; STM from wave 7)", [
                ("LTM alone, step every sample", [f"w8_ltm_nf0.5_every1_s{s}" for s in range(3)], SERIES[1], "-", "o"),
                ("LTM alone, step every 16 samples", [f"w8_ltm_nf0.5_every16_s{s}" for s in range(3)], INK, "-", "o"),
                ("STM (what, r 64, T 4), LTM frozen", [f"w7_what_nf0.5_lr0.01_s{s}" for s in range(3)],
                 SERIES[0], "--", "s")])]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor=SURFACE)
    for ax_row, (title, groups) in zip(axes, rows):
        curves(ax_row, args.run, groups, xmax=200)
        ax_row[0].set_ylim(0, 1.02)
        ax_row[1].set_ylim(0, 1.02)
        ax_row[2].set_ylim(0.8, 1.0)
        ax_row[0].set_ylabel(title, color=INK, fontsize=11)
        ax_row[2].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Wave 8: recency-biased LTM (one step every sample, on the last 16) vs the usual LTM (every 16) vs the "
                 "STM; mean, band = min-max over runs", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, args, "wave8_ltm_vs_stm.png")


def fig_ltm_long(args):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), facecolor=SURFACE)
    groups = [("LTM alone, step every sample", ltm_names(1), SERIES[1], "-", None),
              ("LTM alone, step every 16 samples", ltm_names(16), INK, "-", None)]
    curves(axes[:3], args.run, groups, log=False)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0.7, 1.0)
    # worst of digits 0-8 at each checkpoint
    ax = axes[3]
    for label, names, color, _, _ in groups:
        hs = [hist(args.run, n) for n in names]
        xs = [e["samples"] for e in hs[0]]
        v = np.array([[min(at(h, x, "per_digit")[:9]) for x in xs] for h in hs])
        ax.plot(xs, v.mean(0), color=color, lw=2, label=label)
        ax.fill_between(xs, v.min(0), v.max(0), color=color, alpha=0.12, lw=0)
    style(ax, "Worst single digit of 0-8")
    ax.set_ylim(0.4, 1.0)
    axes[2].legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Wave 8: the two LTM cadences over 3000 samples (25% nines, 5 streams; mean, band = min-max)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave8_ltm_long.png")


def fig_seed_grid(args):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    for a in range(3):
        for b, ls in zip(range(3), ("-", "--", ":")):
            h = hist(args.run, f"w8_grid_seed{a}_ss{b}")
            ax.plot([max(e["samples"], 2) for e in h], [e["new_recall"] for e in h], color=SERIES[a], ls=ls, lw=1.8,
                    label=f"STM seed {a}, stream {b}")
    style(ax, "Held-out nine recall: colour = STM seed, line style = stream")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=3, labelcolor=INK)
    for ax, s in zip(axes[1:], (37, 200)):
        m = np.array([[at(hist(args.run, f"w8_grid_seed{a}_ss{b}"), s, "new_recall") for b in range(3)]
                      for a in range(3)])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", color=INK if m[a, b] < 0.6 else SURFACE)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xlabel("stream seed")
        ax.set_ylabel("STM seed")
        ax.set_title(f"Held-out recall at {s} samples\nvar over STM seed {m.var(0).mean():.4f}, "
                     f"over stream {m.var(1).mean():.4f}", color=INK, loc="left", fontsize=10)
    fig.suptitle("Wave 8: STM seed x stream seed (default what config, 25% nines)", color=INK, x=0.01, ha="left",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave8_seed_grid.png")


def repeat_runs(r, T):
    if (r, T) == (64, 4):
        return [f"w8_grid_seed{s}_ss{s}" for s in range(3)]
    return [f"w8_r{r}_T{T}_s{s}" for s in range(3)]


def fig_repeats(args):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    groups = [(f"r {r}, T {T}", repeat_runs(r, T), SERIES[i], "-" if T == 4 else "--", "o" if T == 4 else "s")
              for i, r in enumerate((16, 64, 128, 256)) for T in (4, 8)]
    for label, names, color, ls, marker in groups:
        xs, mean, _, _ = band(args.run, names)
        for j, ax in enumerate(axes):
            ax.plot(xs, mean[:, j], color=color, lw=2, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1,
                    label=label, zorder=3)
    for ax, t in zip(axes, TITLES):
        style(ax, t)
        ax.set_xscale("log")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0.9, 1.0)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2, labelcolor=INK)
    fig.suptitle("Wave 8: repeats per sample x episode length (T 8 with history 8); mean of 3 seeds (stream = seed)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave8_repeats.png")


def nine_p(run, name):
    """Per few-shot nine: p(9) after training at its last presentation, and the number of presentations."""
    last, pres = {}, {}
    for line in open(os.path.join(run, f"{name}_diag.jsonl")):
        d = json.loads(line)
        if d["new"]:
            last[d["index"]] = d["post"]["p_T"]
            pres[d["index"]] = pres.get(d["index"], 0) + 1
    return last, pres


def reward_runs(run):
    runs = [("accuracy p(y) (default)", ["w8_grid_seed0_ss0", "w8_grid_seed1_ss1", "w8_grid_seed2_ss2"], INK),
            ("log p(y)", ["w8_reward_logp", "w8_reward_logp_s1", "w8_reward_logp_s2"], SERIES[0]),
            ("logit margin", ["w8_reward_margin"], SERIES[1]),
            ("group-norm advantage, 8 repeats", ["w8_group8"], SERIES[2])]
    return [(l, [n for n in names if os.path.exists(os.path.join(run, f"{n}.json"))], c) for l, names, c in runs]


def fig_rewards(args):
    runs = reward_runs(args.run)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    curves(axes[:2], args.run, [(l, n, c, "-" if c == INK else "--", "o") for l, n, c in runs])
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    ax = axes[2]
    xpos = 0
    ticks = []
    for i, (label, names, color) in enumerate(runs):
        for n in names:
            p = np.array(list(nine_p(args.run, n)[0].values()))
            ax.scatter(xpos + np.linspace(-0.25, 0.25, len(p)), np.sort(p), s=22, color=color, edgecolor=SURFACE,
                       lw=0.8, zorder=3)
            ticks.append((xpos, n.replace("w8_", "").replace("grid_", "")))
            xpos += 1
        xpos += 0.5
    ax.axhline(0.5, color=GREY, lw=1, ls=":")
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], rotation=40, ha="right", fontsize=7.5)
    style(ax, "Each few-shot nine: p(9) after its last presentation", xlabel="")
    ax.set_ylim(0, 1.02)
    fig.suptitle("Wave 8: rewards for stuck images (what, r 64, T 4). Dots = the 16 training nines of each run",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave8_rewards.png")


QKV_FAMILIES = [("what", "what (64)", SERIES[0], "-"), ("where", "where (49)", INK, "-"),
                ("v", "v (64)", SERIES[1], "--"), ("q", "q = k (64)", SERIES[2], "--"),
                ("qv", "qv = kv (128)", SERIES[3], ":"), ("qkv", "qkv (192)", SERIES[4], ":")]


def fig_qkv(args):
    res = json.load(open(args.qkv))["families"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    for ax, s in zip(axes, ("train", "held_out")):
        for fam, label, color, ls in QKV_FAMILIES:
            tk = res[fam][s]["topk"]
            ks = sorted(tk, key=int)
            ax.plot([int(k) for k in ks], [np.mean(tk[k]["success"]) for k in ks], color=color, ls=ls, lw=2,
                    marker="o", ms=4.5, mec=SURFACE, mew=1, label=label)
        n = len(res["what"][s]["k_to_flip"])
        style(ax, ("16 training nines" if s == "train" else f"{n} held-out nines the LTM gets wrong")
              + ": fraction flipped to 9", xlabel="bias elements kept (top-k by |log-gain|, re-optimised)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.02)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Wave 8 item 3: offline search (label-supervised, no RL), fewest bias elements to flip an image",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave8_qkv.png")
    return res


def table(args, res):
    R = args.run
    L = ["## LTM cadence vs STM (mean over runs; min-max in brackets)", "",
         "| config | runs | samples | memorised | held-out 9 | digits 0-8 |", "|---|---|---|---|---|---|"]

    def row(label, names, s):
        hs = [hist(R, n) for n in names]
        v = np.array([[at(h, s, k) for k in KEYS] for h in hs], dtype=float)
        L.append(f"| {label} | {len(names)} | {s} | {np.nanmean(v[:, 0]):.2f} | {v[:, 1].mean():.2f} "
                 f"({v[:, 1].min():.2f}-{v[:, 1].max():.2f}) | {v[:, 2].mean():.3f} |")
    for s in (20, 37, 100, 200):
        row("LTM every 1, 25% nines", ltm_names(1), s)
        row("LTM every 16, 25% nines", ltm_names(16), s)
        row("STM default, 25% nines", GRID9, s)
    for s in (500, 1000, 3000):
        row("LTM every 1, 25% nines", ltm_names(1), s)
        row("LTM every 16, 25% nines", ltm_names(16), s)
    for s in (20, 37, 200):
        row("LTM every 1, 50% nines", [f"w8_ltm_nf0.5_every1_s{x}" for x in range(3)], s)
        row("LTM every 16, 50% nines", [f"w8_ltm_nf0.5_every16_s{x}" for x in range(3)], s)
        row("STM, 50% nines (wave 7)", [f"w7_what_nf0.5_lr0.01_s{x}" for x in range(3)], s)
    L += ["", "## Repeats x T (3 seeds): held-out recall", "", "| config | 10 | 20 | 37 | 52 | 100 | 200 | "
          "digits 0-8 at 200 | nines < 0.5 at last presentation (3 seeds) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in (16, 64, 128, 256):
        for T in (4, 8):
            names = repeat_runs(r, T)
            hs = [hist(R, n) for n in names]
            stuck = sum(int((np.array(list(nine_p(R, n)[0].values())) < 0.5).sum()) for n in names)
            L.append(f"| r {r} T {T} | " + " | ".join(f"{np.mean([at(h, s, 'new_recall') for h in hs]):.2f}"
                                                   for s in (10, 20, 37, 52, 100, 200))
                     + f" | {np.mean([at(h, 200, 'others_acc') for h in hs]):.3f} | {stuck}/48 |")
    L += ["", "## Rewards (per run)", "", "| run | held-out 37 | held-out 200 | memorised 200 | digits 0-8 200 | "
          "nines < 0.5 | worst p(9) (index) |", "|---|---|---|---|---|---|---|"]
    for label, names, _ in reward_runs(R):
        for n in names:
            h = hist(R, n)
            p, _ = nine_p(R, n)
            w = min(p, key=p.get)
            L.append(f"| {label}: {n} | {at(h, 37, 'new_recall'):.2f} | {at(h, 200, 'new_recall'):.2f} | "
                     f"{at(h, 200, 'new_train_recall'):.2f} | {at(h, 200, 'others_acc'):.3f} | "
                     f"{sum(v < 0.5 for v in p.values())}/16 | {p[w]:.2f} ({w}) |")
    L += ["", "## Q/K/V search", "", "| family | size | set | dense success | flipped | median k | success k<=8 | "
          "median sum abs log-gain (dense) |", "|---|---|---|---|---|---|---|---|"]
    for fam, d in res.items():
        for s in ("train", "held_out"):
            kf = [k for k in d[s]["k_to_flip"] if k is not None]
            L.append(f"| {fam} | {d['size']} | {s} | {np.mean(d[s]['dense']['success']):.2f} | "
                     f"{len(kf)}/{len(d[s]['k_to_flip'])} | {np.median(kf):.0f} | "
                     f"{np.mean(d[s]['topk']['8']['success']):.2f} | {np.median(d[s]['dense']['abs_gain_sum']):.1f} |")
    path = os.path.join(args.out, "wave8_table.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(open(path).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--qkv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig_ltm_vs_stm(args)
    fig_ltm_long(args)
    fig_seed_grid(args)
    fig_repeats(args)
    fig_rewards(args)
    table(args, fig_qkv(args))


if __name__ == "__main__":
    main()
