"""
Learning speed in samples on the frozen gap snapshot: digit-9 recall and other digits' accuracy vs samples presented,
for the LTM trained on the stream (stm_frozen ltm) and the RL STM (stm_frozen rl), same stream and checkpoints.

  python -m single_stream.plot_stm_frozen --run runs_ss/stm_frozen_20261101/snap9 --out <dir>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference categorical palette, slots in fixed order (dataviz skill references/palette.md)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

RUNS = [  # (file, label, is_ltm)
    ("ltm_lr0.001", "LTM (AdamW 1e-3)", True),
    ("ltm_lr0.0003", "LTM (AdamW 3e-4)", True),
    ("w2_where_term", "STM where, terminal, actor 0.01", False),
    ("w2_where_term_a0.1", "STM where, terminal, actor 0.1", False),
    ("w2_where_impr_a0.1", "STM where, improvement, actor 0.1", False),
    ("w2_what_term_a0.1", "STM what, terminal, actor 0.1", False),
]


def load(run, name):
    hist = json.load(open(os.path.join(run, f"{name}.json")))["result"]["history"]
    xs = [h.get("samples", h.get("episodes")) for h in hist]
    return xs, [h["new_recall"] for h in hist], [h["others_acc"] for h in hist], hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    table = []
    for i, (name, label, is_ltm) in enumerate(RUNS):
        xs, new, others, hist = load(args.run, name)
        xs = [max(x, 30) for x in xs]  # sample 0 drawn at the left edge of the log axis
        for ax, ys in ((axes[0], new), (axes[1], others)):
            ax.plot(xs, ys, color=SERIES[i], lw=2, ls="-" if is_ltm else "--", marker="o" if is_ltm else "s",
                    ms=5, mec=SURFACE, mew=1.5, label=label, zorder=3)
        table.append((label, new[-1], others[-1]))
    for ax, title in ((axes[0], "Digit-9 recall (the few-shot gap)"), (axes[1], "Accuracy on digits 0-8")):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_xlabel("samples presented (25% are the 16 few-shot nines)")
        ax.set_title(title, color=INK, loc="left")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("recall on validation nines")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].set_ylabel("accuracy")
    axes[0].annotate("unbiased snapshot 0.10;\noptimal bias (oracle) 0.97", xy=(30, 0.10), xytext=(1500, 0.42),
                     color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    axes[0].axhline(0.972, color=INK2, lw=1, ls=":", zorder=1)
    axes[0].text(25000, 0.95, "oracle bias", color=INK2, fontsize=8, ha="right", va="top")
    axes[1].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Frozen gap snapshot: LTM vs RL STM learning speed, measured in samples", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    path = os.path.join(args.out, "learning_speed_samples.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    with open(os.path.join(args.out, "learning_speed_samples.md"), "w") as f:
        f.write("| run | digit-9 recall at 20k samples | digits 0-8 acc at 20k |\n|---|---|---|\n")
        for label, n, o in table:
            f.write(f"| {label} | {n:.3f} | {o:.4f} |\n")
    print(path)


if __name__ == "__main__":
    main()
