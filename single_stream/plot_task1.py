"""
Task-1 round 1-3 figures (permuted-MNIST stage, MNIST unpermuted, Dohare MLP, seed 0, validation split).

  python -m single_stream.plot_task1 --out ~/dev/claude/single_stream_paper/figures/task1_r123

Fig 1 vs cluster size: final, anytime, worst grid checkpoint after epoch 1, worst digit at the end.
Fig 2 accuracy over the stream, one panel per cluster size.
Fig 3 sensitivity at c = 128: continual-backprop replacement rate; step size.
Fig 4 balanced few-shot exposure efficiency and retention (bias-proof per-digit score), c = 64 and 128.
Missing runs are skipped (the script also works on a partial set).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "runs_ss")
R1S, R1 = "task1_r1_select_20261024", "task1_r1_csweep_20261024"
R2, R3S, R3 = "task1_r2_metrics_20261024", "task1_r3_replay_select_20261024", "task1_r3_replay_csweep_20261025"
EPOCH = 54000
PLAIN_T = 3 * EPOCH  # plain and continual-backprop budget (3 epochs); replay-only selected 5

# Reference palette, light mode (dataviz skill): slots 1-3 validate all-pairs.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# model -> (label, colour, {c: run dir relative to ROOT}, budget in exposures)
MODELS = {
    "plain": ("LTM-only", BLUE, {1: f"{R2}/mlp_c1_lr0.003_s0", 16: f"{R1}/mlp_c16_lr0.003_s0",
                                 64: f"{R2}/mlp_c64_lr0.003_s0", 128: f"{R2}/mlp_c128_lr0.003_s0"}, PLAIN_T),
    "cbp": ("LTM + continual backprop", ORANGE,
            {1: f"{R2}/mlpcbp_c1_lr0.003_s0", 16: f"{R1}/mlpcbp_c16_lr0.003_s0",
             64: f"{R2}/mlpcbp_c64_lr0.003_s0", 128: f"{R2}/mlpcbp_c128_lr0.003_s0"}, PLAIN_T),
    "replay": ("Replay-only LTM, FIFO 1000", AQUA,
               {1: f"{R3S}/ro_fifo1000_c1_lr0.1_s0", 16: f"{R3}/ro_fifo1000_c16_lr0.1_s0",
                64: f"{R3}/ro_fifo1000_c64_lr0.1_s0", 128: f"{R3}/ro_fifo1000_c128_lr0.1_s0"}, 5 * EPOCH),
}
# univariate variants at c = 128, drawn in the colour of the model they vary, hollow marker
VARIANTS = {
    "cbp_eta": ("continual backprop, utility decay 0.999", ORANGE, f"{R2}/mlpcbp_eta0.999_c128_lr0.003_s0",
                PLAIN_T),
    "fifo4000": ("replay-only, FIFO 4000", AQUA, f"{R3}/ro_fifo4000_c128_lr0.1_s0", 5 * EPOCH),
}
CS = [1, 16, 64, 128]


def load(rel):
    d = os.path.join(ROOT, rel)
    if not os.path.exists(os.path.join(d, "summary.json")):
        return None
    with open(os.path.join(d, "checkpoints.jsonl")) as f:
        records = [json.loads(line) for line in f]
    with open(os.path.join(d, "summary.json")) as f:
        summary = json.load(f)
    return records, summary


def metrics(run, tmax):
    records, _ = run
    recs = [r for r in records if r["t"] <= tmax]
    grid = [r for r in recs if r["grid"]]
    late = [r["acc"] for r in grid if r["t"] > EPOCH]
    return {"final": recs[-1]["acc"], "anytime": float(np.mean([r["acc"] for r in grid])),
            "worst": min(late) if late else np.nan, "worst_digit": min(recs[-1]["group_acc"])}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def cluster_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(CS)
    ax.set_xticklabels([str(c) for c in CS])
    ax.set_xlabel("cluster size c (log scale)")


def fig_vs_cluster(out):
    panels = [("final", "Final accuracy"), ("anytime", "Anytime accuracy"),
              ("worst", "Worst checkpoint after epoch 1"), ("worst_digit", "Worst digit at the end")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        for name, (label, colour, runs, budget) in MODELS.items():
            pts = [(c, metrics(r, budget)[key]) for c in CS if (r := load(runs[c]))]
            if pts:
                ax.plot(*zip(*pts), color=colour, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label)
            if name == "replay":  # like-for-like at the plain budget
                pts = [(c, metrics(r, PLAIN_T)[key]) for c in CS if (r := load(runs[c]))]
                if pts:
                    ax.plot(*zip(*pts), color=colour, lw=1.5, ls="--", marker="o", ms=7, mfc=SURFACE,
                            mec=colour, mew=1.5, label="Replay-only at 3 epochs (162k exposures)")
        for name, (label, colour, rel, budget) in VARIANTS.items():
            r = load(rel)
            if r:
                ax.plot([128], [metrics(r, budget)[key]], ls="none", marker="D", ms=8, mfc=SURFACE, mec=colour,
                        mew=2, label=label)
        cluster_axis(ax)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("validation accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Task 1, MNIST, Dohare MLP 3x2000, b = 1, seed 0: effect of cluster size", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(os.path.join(out, "fig1_vs_cluster_size.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_over_stream(out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True, facecolor=SURFACE)
    for ax, c in zip(axes, CS):
        style(ax)
        for name, (label, colour, runs, budget) in MODELS.items():
            r = load(runs[c])
            if r:
                ts = [x["t"] / EPOCH for x in r[0]]
                ax.plot(ts, [x["acc"] for x in r[0]], color=colour, lw=1.6, label=label)
        if c == 128:
            for name, (label, colour, rel, budget) in VARIANTS.items():
                r = load(rel)
                if r:
                    ax.plot([x["t"] / EPOCH for x in r[0]], [x["acc"] for x in r[0]], color=colour, lw=1.4,
                            ls="--", label=label)
        ax.set_title(f"c = {c}", fontsize=11)
        ax.set_xlabel("epochs (54k exposures each)")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 5.1)
    axes[0].set_ylabel("validation accuracy")
    handles, labels = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Accuracy over the stream (checkpoints; replay-only runs 5 epochs, the others 3)", color=INK,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(out, "fig2_over_stream.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_sensitivity(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    ax = axes[0]
    style(ax)
    rates = [(1e-6, f"{R2}/mlp_c128_lr0.003_s0"), (1e-5, f"{R1}/mlpcbp1e-5_c128_lr0.003_s0"),
             (1e-4, f"{R2}/mlpcbp_c128_lr0.003_s0"), (1e-3, f"{R1}/mlpcbp1e-3_c128_lr0.003_s0")]
    for key, label, ls in (("final", "final", "-"), ("anytime", "anytime", "--")):
        pts = [(x, metrics(r, PLAIN_T)[key]) for x, rel in rates if (r := load(rel))]
        if pts:
            ax.plot(*zip(*pts), color=ORANGE, lw=2, ls=ls, marker="o", ms=7, mec=SURFACE, mew=1.5, label=label)
    ax.set_xscale("log")
    ax.set_xticks([x for x, _ in rates])
    ax.set_xticklabels(["0 (plain)", "1e-5", "1e-4", "1e-3"])
    ax.set_xlabel("continual-backprop replacement rate")
    ax.set_ylabel("validation accuracy")
    ax.set_title("c = 128: replacement rate (decay 0.99, lr 0.003)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax = axes[1]
    style(ax)
    for tag, label, colour, first in (("mlp", "LTM-only", BLUE, R2), ("mlpcbp", "LTM + continual backprop", ORANGE, R2)):
        for key, ls in (("final", "-"), ("anytime", "--")):
            pts = []
            for lr in (0.0003, 0.001, 0.003):
                rel = f"{first}/{tag}_c128_lr0.003_s0" if lr == 0.003 else f"{R1}/{tag}_c128_lr{lr}_s0"
                r = load(rel)
                if r:
                    pts.append((lr, metrics(r, PLAIN_T)[key]))
            if pts:
                ax.plot(*zip(*pts), color=colour, lw=2, ls=ls, marker="o", ms=7, mec=SURFACE, mew=1.5,
                        label=f"{label}, {key}")
    ax.set_xscale("log")
    ax.set_xticks([0.0003, 0.001, 0.003])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticklabels(["0.0003", "0.001", "0.003 (selected)"])
    ax.set_xlabel("step size")
    ax.set_title("c = 128: step size", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_sensitivity_c128.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_fewshot_retention(out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, facecolor=SURFACE)
    for col, c in enumerate((64, 128)):
        for row, (key, xlabel) in enumerate((("exposure_efficiency_balanced", "exposures of the digit so far"),
                                             ("retention_balanced", "exposures since the digit was last seen"))):
            ax = axes[row, col]
            style(ax)
            for name, (label, colour, runs, budget) in MODELS.items():
                r = load(runs[c])
                if r and key in r[1]:
                    pts = [(lo, acc) for lo, acc, n in r[1][key] if lo > 0]
                    ax.plot(*zip(*pts), color=colour, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2, label=label)
            ax.axhline(0.5, color=INK2, lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xlabel(xlabel + " (log2 bins, lower edge)")
            ax.set_title(f"c = {c}: " + ("few-shot exposure efficiency" if row == 0 else "retention"), fontsize=11)
            ax.set_ylim(0.4, 1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("balanced per-digit score (chance 0.5)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Bias-proof per-digit score, pooled over checkpoints (seed 0)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(out, "fig4_fewshot_retention.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def table(out):
    rows = ["| model | c | budget | final | anytime | worst after ep 1 | worst digit | final @162k | anytime @162k |",
            "|---|---|---|---|---|---|---|---|---|"]
    entries = [(label, c, runs[c], budget) for name, (label, colour, runs, budget) in MODELS.items() for c in CS]
    entries += [(label, 128, rel, budget) for name, (label, colour, rel, budget) in VARIANTS.items()]
    for label, c, rel, budget in entries:
        r = load(rel)
        if not r:
            rows.append(f"| {label} | {c} | pending | | | | | | |")
            continue
        m, m3 = metrics(r, budget), metrics(r, PLAIN_T)
        rows.append(f"| {label} | {c} | {budget // EPOCH} ep | {m['final']:.4f} | {m['anytime']:.4f} | "
                    f"{m['worst']:.3f} | {m['worst_digit']:.3f} | {m3['final']:.4f} | {m3['anytime']:.4f} |")
    with open(os.path.join(out, "table_task1.md"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig_vs_cluster(a.out)
    fig_over_stream(a.out)
    fig_sensitivity(a.out)
    fig_fewshot_retention(a.out)
    print(table(a.out))


if __name__ == "__main__":
    main()
