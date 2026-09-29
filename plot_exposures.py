"""
Re-plot continual-learning accuracy against a common exposure axis (exposure efficiency): the number of
training-image exposures (one exposure = one image shown once; an STM episode of 8 steps on one image counts once).

Why: the reference scripts both log an "epoch" index, but the two loops mean different things by it.
  - cifar_main_ltm_fine_tuning.py (and the head baselines): one epoch = one shuffled pass over the phase's
    training set (1000 images for a coarse pair at 500 instances per class).
  - cifar_main_stm_training.py: one "epoch" = TRAINING_STEPS (4000) vector steps of BATCH_SIZE (16) parallel
    environments, each environment running 8-step episodes on images drawn uniformly at random with
    replacement. That is 4000 * 16 / 8 = 8000 presentations per reported epoch, i.e. 8 passes' worth.
So the paper's shared "Epoch" axis gives the STM 8x the presentations of the LTM at every tick.

Usage: python plot_exposures.py [--runs-root archive_e40] [--out-dir <dir>]
"""
import argparse, glob, os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_comparison import sniff_reference_kind, parse_order, parse_seed, load_pivot, FINE_NAMES  # noqa: E402

ORDERS = [(3, 4, 5), (4, 5, 3), (5, 3, 4)]
HEADS = ["linear", "ncm", "flymodel", "sdmlp"]
LABEL = {"stm": "CLS/STM", "ltm": "LTM-only", "linear": "linear head", "ncm": "NCM head",
         "flymodel": "FlyModel head", "sdmlp": "SDMLP head"}
COLORS = {"12": "tab:blue", "3": "tab:orange", "4": "tab:red", "5": "tab:purple"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="archive_e40")
    p.add_argument("--out-dir", default="runs_local/exposures")
    p.add_argument("--stm-steps", type=int, default=4000, help="TRAINING_STEPS per reported STM epoch")
    p.add_argument("--stm-envs", type=int, default=16, help="BATCH_SIZE = parallel environments")
    p.add_argument("--episode-steps", type=int, default=8)
    p.add_argument("--train-size", type=int, default=1000, help="training images per phase (2 coarse x 500)")
    p.add_argument("--epochs-per-phase", type=int, default=12)
    return p.parse_args()


def load_reference(root):
    out = {"stm": {}, "ltm": {}}
    for f in glob.glob(os.path.join(root, "runs/cifar_100/continual_[[]*[]]_500/**/results_continual.txt"),
                       recursive=True):
        kind = sniff_reference_kind(f)
        order = tuple(parse_order(f) or ())
        if kind in out and order in ORDERS:
            piv = load_pivot(f)
            if piv is not None and len(piv) == 36:
                out[kind].setdefault(order, []).append(piv)
    return out


def load_heads(root):
    out = {h: {} for h in HEADS}
    for h in HEADS:
        for f in glob.glob(os.path.join(root, f"runs/cifar_100/head_{h}_continual_[[]*[]]_500/seed_*/results_continual.txt")):
            order = tuple(parse_order(f) or ())
            piv = load_pivot(f)
            if piv is not None and order in ORDERS and len(piv) == 36:
                out[h].setdefault(order, []).append(piv)
    return out


def mean_pivot(pivs):
    arr = np.stack([p[FINE_NAMES].to_numpy(float) for p in pivs])
    return pd.DataFrame(arr.mean(0), index=pivs[0].index, columns=FINE_NAMES), len(pivs)


def expected_unique(n, N, with_replacement):
    return N * (1 - (1 - 1 / N) ** n) if with_replacement else min(n, N)


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    per_epoch = {"stm": a.stm_steps * a.stm_envs // a.episode_steps, "ltm": a.train_size}
    for h in HEADS:
        per_epoch[h] = a.train_size
    E = a.epochs_per_phase

    data = load_reference(a.runs_root)
    data.update(load_heads(a.runs_root))
    methods = [m for m in ["stm", "ltm"] + HEADS if data.get(m)]
    means = {m: {o: mean_pivot(data[m][o]) for o in ORDERS if o in data[m]} for m in methods}

    def curve(m, order, phase, col):
        """(presentations within phase, accuracy on test set col) at each evaluation in the phase."""
        piv, _ = means[m][order]
        rows = piv.iloc[phase * E:(phase + 1) * E]
        x = np.arange(1, E + 1) * per_epoch[m]
        return x, rows[col].to_numpy()

    # ---------------- Figure 1: per phase, trained class, STM vs LTM, common x ----------------
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
    for r, order in enumerate(ORDERS):
        for phase in range(3):
            ax = axes[r, phase]
            fc = str(order[phase])
            others = [c for c in FINE_NAMES if c != fc]
            for m, ls, lw in [("stm", "-", 2.2), ("ltm", "--", 2.2)]:
                if order not in means[m]:
                    continue
                x, y = curve(m, order, phase, fc)
                ax.plot(x, y, ls, color=COLORS[fc], lw=lw, marker="o", ms=4, label=f"{LABEL[m]}: class {fc} (trained)")
                yo = np.mean([curve(m, order, phase, c)[1] for c in others], axis=0)
                ax.plot(x, yo, ls, color="grey", lw=1.4, marker=".", ms=3, label=f"{LABEL[m]}: mean of other test sets")
            ax.axvline(E * per_epoch["ltm"], color="k", lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_xlim(800, E * per_epoch["stm"] * 1.2)
            ax.axhline(0.5, color="k", lw=0.8, ls="-.")
            ax.grid(alpha=0.3, which="both")
            ax.set_title(f"order {','.join(map(str, order))}  |  phase {phase + 1}: train fine-class {fc}", fontsize=11)
            if r == 2:
                ax.set_xlabel("training-image exposures within phase (log)")
            if phase == 0:
                ax.set_ylabel("test accuracy")
            if r == 0 and phase == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Accuracy vs exposures (one exposure = one image shown once, however many STM steps or updates occur). LTM-only: 1 epoch = {per_epoch['ltm']} exposures, 12 epochs = "
                 f"{E * per_epoch['ltm']:,} (dotted line). CLS/STM: 1 reported 'epoch' = {a.stm_steps} steps x "
                 f"{a.stm_envs} envs / {a.episode_steps} steps per episode = {per_epoch['stm']:,} exposures, "
                 f"12 'epochs' = {E * per_epoch['stm']:,}.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f1 = os.path.join(a.out_dir, "exposures_per_phase.png")
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # ---------------- Figure 2: whole run, cumulative presentations, overlay ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, order in enumerate(ORDERS):
        ax = axes[i]
        for m, ls in [("stm", "-"), ("ltm", "--")]:
            if order not in means[m]:
                continue
            piv, _ = means[m][order]
            x = np.arange(1, 3 * E + 1) * per_epoch[m]
            for c in FINE_NAMES:
                ax.plot(x, piv[c].to_numpy(), ls, color=COLORS[c], lw=1.6, marker="o", ms=2.5,
                        label=f"{LABEL[m]} {c}")
            for b in (E, 2 * E):
                ax.axvline(b * per_epoch[m], color="k" if m == "stm" else "grey", lw=0.8, ls=ls)
        ax.set_xscale("log")
        ax.axhline(0.5, color="k", lw=0.8, ls="-.")
        ax.grid(alpha=0.3, which="both")
        ax.set_title(f"order {','.join(map(str, order))}  (solid = CLS/STM, dashed = LTM-only; "
                     f"vertical lines = phase boundaries)", fontsize=10)
        ax.set_xlabel("cumulative training-image exposures")
        if i == 0:
            ax.set_ylabel("test accuracy")
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    f2 = os.path.join(a.out_dir, "exposures_cumulative.png")
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    # ---------------- Tables ----------------
    lines = []
    lines.append(f"# Accuracy vs training-image exposures (exposure efficiency) (runs root: {a.runs_root})\n")
    lines.append(f"Exposures per reported epoch (one exposure = one image shown once, regardless of the 8 STM steps and updates on it): CLS/STM {per_epoch['stm']:,} ({a.stm_steps} steps x {a.stm_envs} "
                 f"envs / {a.episode_steps} steps per episode, images sampled with replacement); LTM-only and heads "
                 f"{per_epoch['ltm']:,} (one shuffled pass). Training set per phase: {a.train_size} unique images.\n")
    n_stm = int(per_epoch["stm"])
    lines.append("Expected number of UNIQUE training images seen within a phase after n exposures "
                 f"(STM samples with replacement; LTM is a shuffled pass): "
                 f"STM after {n_stm:,} = {expected_unique(n_stm, a.train_size, True):.1f}; "
                 f"LTM after {a.train_size:,} = {expected_unique(a.train_size, a.train_size, False)}. "
                 "Both models have seen essentially every unique image by their first evaluation, so a "
                 "unique-image axis is flat after the first tick; the tables use exposures.\n")

    # Table A: trained-class accuracy at matched exposure budgets
    lines.append("## A. Accuracy on the class being trained, at matched exposure budgets\n")
    lines.append("LTM at 8k = its epoch 8; LTM at 12k = its epoch 12 (end of phase). STM at 8k = its reported epoch 1; "
                 "STM at 16k = reported epoch 2; STM at 96k = reported epoch 12 (end of phase). Heads: mean over seeds.\n")
    hdr = "| order | phase (class) | " + " | ".join(
        [f"{LABEL[m]} @8k" for m in methods] + [f"{LABEL[m]} @12k" for m in methods if m != "stm"] +
        ["CLS/STM @16k", "CLS/STM @96k"]) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (hdr.count("|") - 1))
    for order in ORDERS:
        for phase in range(3):
            fc = str(order[phase])
            cells = []
            for m in methods:
                x, y = curve(m, order, phase, fc)
                cells.append(f"{y[np.searchsorted(x, 8000)]:.3f}" if order in means[m] and 8000 in x else "n/a")
            for m in methods:
                if m == "stm":
                    continue
                x, y = curve(m, order, phase, fc)
                cells.append(f"{y[np.searchsorted(x, 12000)]:.3f}" if 12000 in x else "n/a")
            x, y = curve("stm", order, phase, fc)
            cells.append(f"{y[1]:.3f}")
            cells.append(f"{y[-1]:.3f}")
            lines.append(f"| {','.join(map(str, order))} | {phase + 1} ({fc}) | " + " | ".join(cells) + " |")
    lines.append("")

    # Table B: presentations needed to first reach thresholds on the trained class
    lines.append("## B. Exposures needed for the trained class to first reach a threshold (first evaluation at or above)\n")
    lines.append("Evaluations happen every 1,000 exposures for LTM/heads and every 8,000 for the STM, so STM values are "
                 "upper bounds with 8,000 granularity. 'never' = not reached within the phase.\n")
    for thr in (0.60, 0.70, 0.80):
        lines.append(f"### threshold {thr:.2f}\n")
        lines.append("| order | phase (class) | " + " | ".join(LABEL[m] for m in methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 2))
        for order in ORDERS:
            for phase in range(3):
                fc = str(order[phase])
                cells = []
                for m in methods:
                    if order not in means[m]:
                        cells.append("n/a"); continue
                    x, y = curve(m, order, phase, fc)
                    hit = np.where(y >= thr)[0]
                    cells.append(f"{x[hit[0]]:,}" if len(hit) else "never")
                lines.append(f"| {','.join(map(str, order))} | {phase + 1} ({fc}) | " + " | ".join(cells) + " |")
        lines.append("")

    # Table C: retention at end of phase (mean over the three test sets not being trained)
    lines.append("## C. Mean accuracy on the three test sets NOT being trained, at end of each phase\n")
    lines.append("| order | phase (class) | " + " | ".join(LABEL[m] for m in methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 2))
    for order in ORDERS:
        for phase in range(3):
            fc = str(order[phase])
            others = [c for c in FINE_NAMES if c != fc]
            cells = []
            for m in methods:
                if order not in means[m]:
                    cells.append("n/a"); continue
                vals = [curve(m, order, phase, c)[1][-1] for c in others]
                cells.append(f"{np.mean(vals):.3f}")
            lines.append(f"| {','.join(map(str, order))} | {phase + 1} ({fc}) | " + " | ".join(cells) + " |")
    lines.append("")
    seeds = {m: {o: means[m][o][1] for o in means[m]} for m in methods}
    lines.append(f"Seeds per method: " + ", ".join(f"{LABEL[m]} {min(seeds[m].values())}" for m in methods) + ".\n")

    f3 = os.path.join(a.out_dir, "exposures_tables.md")
    with open(f3, "w") as f:
        f.write("\n".join(lines))
    print("wrote", f1, f2, f3)


if __name__ == "__main__":
    main()
