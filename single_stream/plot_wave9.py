"""
Wave 9 figures (thinking time; journal 3 Nov 2026): adaptive repeats, sigma boost, prioritised replay vs the fixed-repeat
references, against the LTM envelope (wave 8). 5 STM seeds, stream = seed; references for seeds 0-2 are wave-8 runs.

  wave9_samples.png    memorised (training nines shown so far) / held-out / digits 0-8 vs nines presented (log x),
                       with LTM every 1 and every 16 (held-out and digits 0-8 only: the LTM runs do not log shown-only)
  wave9_episodes.png   memorised (shown so far) and held-out vs learning episodes (thinking time), log x
  wave9_table.md       recall at matched samples, learning episodes, repeats per presentation, first-presentation success

  python -m single_stream.plot_wave9 --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import collections
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import TITLES, hist, ltm_names, save, style

# memorised = recall on the training nines shown so far (mem_seen; None before a run's first nine, so the seed mean
# starts at the first checkpoint where every seed has shown one: 9 samples); the LTM runs do not log it, so they
# appear in the held-out and digits 0-8 panels only
def at(h, s, k):
    """Value of key k at the last checkpoint <= s samples (None if absent)."""
    c = [e for e in h if e["samples"] <= s]
    return c[-1].get(k) if c else np.nan


KEYS = ("mem_seen", "new_recall", "others_acc")
TITLES = ("Memorised: recall on the training nines shown so far",) + TITLES[1:]


# x axis (Dave, 3 Nov): few-shot cost = nine presentations from the stream (new_samples), not samples of all digits
# and not replay. Each run is linearly interpolated between its checkpoints onto this grid (seeds reach a nine count at
# different sample counts); every STM run has >= 43 nines by sample 200.
NINES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)


def on_nines(h, k, nines=NINES):
    """Run h's key k at the given nine counts (NaN where absent, not yet defined, or beyond the run)."""
    ns = np.array([e["new_samples"] for e in h], dtype=float)
    v = np.array([np.nan if e.get(k) is None else e[k] for e in h], dtype=float)
    out = np.interp(nines, ns, v)
    out[np.asarray(nines) > ns[-1]] = np.nan
    return out


def band(run, names):
    """Seed mean of KEYS on the NINES grid (NaN where any run has no value)."""
    hs = [hist(run, n) for n in names]
    v = np.array([np.stack([on_nines(h, k) for k in KEYS], 1) for h in hs])
    return np.array(NINES), v.mean(0)

REF = {"r 64 (reference)": [f"w8_grid_seed{s}_ss{s}" for s in range(3)] + [f"w9_r64_s{s}" for s in (3, 4)],
       "r 256 T 4 (reference)": [f"w8_r256_T4_s{s}" for s in range(3)] + [f"w9_r256_T4_s{s}" for s in (3, 4)],
       "r 256 T 8 (reference)": [f"w8_r256_T8_s{s}" for s in range(3)] + [f"w9_r256_T8_s{s}" for s in (3, 4)]}
NEW = {"r 64 + boost": "boost_r64", "r 64 + prio": "prio_r64", "adapt 256": "adapt256",
       "adapt 256 + boost": "adapt256_boost", "adapt 512 + boost": "adapt512_boost",
       "adapt 256 + boost, T 8": "adapt256_boost_T8", "adapt 256 + boost + prio": "adapt256_boost_prio"}
CONFIGS = {**REF, **{k: [f"w9_{v}_s{s}" for s in range(5)] for k, v in NEW.items()}}
TABLE_NINES = (2, 5, 9, 16, 40)
NINES_TICKS = (1, 2, 5, 10, 20, 40)
# (label, color, linestyle, marker) for the figures; the rest are in the table only
SHOWN = [("r 64 (reference)", INK, "--", "s"), ("r 256 T 4 (reference)", SERIES[2], "--", "s"),
         ("adapt 256", SERIES[0], "-", "o"), ("adapt 256 + boost", SERIES[3], "-", "o"),
         ("adapt 256 + boost + prio", SERIES[4], "-", "o")]
LTM = [("LTM alone, step every sample", ltm_names(1), SERIES[1], ":", None),
       ("LTM alone, step every 16 samples", ltm_names(16), "#8f8d87", ":", None)]


def presentations(run, name, max_sample=200):
    """Per nine presentation: repeats, whether sigma was boosted, and p(y) >= 0.5 after it; first presentations."""
    rows, seen = [], collections.Counter()
    fixed = json.load(open(os.path.join(run, f"{name}.json")))["args"]["repeats"]  # wave-8 diag has no repeats field
    for line in open(os.path.join(run, f"{name}_diag.jsonl")):
        d = json.loads(line)
        if d["sample"] > max_sample:
            continue
        seen[d["index"]] += d["new"]
        rows.append({"new": d["new"], "first": d["new"] and seen[d["index"]] == 1, "repeats": d.get("repeats", fixed),
                     "boosted": d.get("sigma_max_used", 0.5) > 0.5, "right": d["post"]["p_T"] >= 0.5})
    return rows


def fig_samples(args):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    for label, names, color, ls, marker in LTM + [(l, CONFIGS[l], c, ls, m) for l, c, ls, m in SHOWN]:
        xs, mean = band(args.run, names)
        for j, ax in enumerate(axes):
            ax.plot(xs, mean[:, j], color=color, lw=2, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1,
                    label=label, zorder=3)
    for ax, t in zip(axes, TITLES):
        style(ax, t, xlabel="nines presented")
        ax.set_xscale("log")
        ax.set_xticks(NINES_TICKS, [str(n) for n in NINES_TICKS])
        ax.set_xlim(0.9, 44)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[2].set_ylim(0.85, 1.0)
    axes[2].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Wave 9: adaptive repeats (block 8, stop at p 0.8 or 4 flat blocks), sigma boost, prioritised replay; "
                 "mean of 5 STM seeds (stream = seed)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "wave9_samples.png")


def fig_episodes(args):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for label, color, ls, marker in SHOWN:
        hs = [hist(args.run, n) for n in CONFIGS[label]]
        n = min(len(h) for h in hs)  # checkpoints (shared schedule); mean over seeds, NaN until every seed has a value
        eps = np.mean([[e["learning_episodes"] for e in h[:n]] for h in hs], 0)
        for j, ax in enumerate(axes):
            v = np.mean([[np.nan if e.get(KEYS[j]) is None else e[KEYS[j]] for e in h[:n]] for h in hs], 0)
            ax.plot(np.maximum(eps, 1), v, color=color, lw=2, ls=ls, marker=marker, ms=4, mec=SURFACE, mew=1,
                    label=label, zorder=3)
    for ax, t in zip(axes, TITLES[:2]):
        style(ax, t, xlabel="learning episodes (thinking time), mean over seeds")
        ax.set_xlim(0.6, 2e5)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Wave 9: recall against thinking time; markers are the checkpoints (2 ... 200 samples)", color=INK,
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, args, "wave9_episodes.png")


def table(args):
    t = " / ".join(map(str, TABLE_NINES))
    L = ["## Wave 9 vs references and the LTM envelope, at nines presented (mean of 5 seeds; held-out min-max at 9 nines "
         "in brackets; digits 0-8 and learning episodes at 200 samples of all digits)", "",
         f"| config | memorised, shown so far {t} | all 16 training nines {t} | held-out {t} "
         "| digits 0-8 @200 samples | learning episodes @200 samples |", "|---|---|---|---|---|---|"]
    for label, names in [(l, n) for l, n, *_ in LTM] + list(CONFIGS.items()):
        hs = [hist(args.run, n) for n in names]
        m = lambda k, s: np.nanmean([np.nan if at(h, s, k) is None else at(h, s, k) for h in hs])
        mn = lambda k: np.nanmean([on_nines(h, k, TABLE_NINES) for h in hs], 0)
        fmt = lambda v: " / ".join(f"{x:.2f}" for x in v)
        h9 = [float(on_nines(h, "new_recall", (9,))[0]) for h in hs]
        eps = f"{np.mean([at(h, 200, 'learning_episodes') for h in hs]):.0f}" if "learning_episodes" in hs[0][-1] \
            else "-"
        L.append(f"| {label} | " + (fmt(mn("mem_seen")) if "mem_seen" in hs[0][-1] else "-") + " | "
                 + fmt(mn("new_train_recall")) + " | " + fmt(mn("new_recall"))
                 + f" ({min(h9):.2f}-{max(h9):.2f}) | {m('others_acc', 200):.3f} | {eps} |")
    L += ["", "## Nine presentations up to sample 200 (5 seeds, 16 nines each)", "",
          "| config | repeats per nine pres (mean / median) | repeats per other sample | right after 1st pres | "
          "nines never right | boosted pres | boosted pres ending right |", "|---|---|---|---|---|---|---|"]
    for label, names in CONFIGS.items():
        rows = [r | {"run": n} for n in names for r in presentations(args.run, n)]
        nine = [r for r in rows if r["new"]]
        ever = collections.defaultdict(set)
        for n in names:
            for line in open(os.path.join(args.run, f"{n}_diag.jsonl")):
                d = json.loads(line)
                if d["new"] and d["sample"] <= 200:
                    if d["post"]["p_T"] >= 0.5:
                        ever[n].add(d["index"])
        never = sum(16 - len(ever[n]) for n in names)
        rn = [r["repeats"] for r in nine]
        ro = [r["repeats"] for r in rows if not r["new"]]
        b = [r for r in nine if r["boosted"]]
        L.append(f"| {label} | {np.mean(rn):.1f} / {np.median(rn):.0f} | {np.mean(ro):.1f} | "
                 f"{np.mean([r['right'] for r in nine if r['first']]):.2f} | {never}/{16 * len(names)} | "
                 f"{len(b)}/{len(nine)} | {sum(r['right'] for r in b)}/{len(b)} |")
    path = os.path.join(args.out, "wave9_table.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_samples(args)
    fig_episodes(args)
    table(args)


if __name__ == "__main__":
    main()
