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


def load_log(run, name):
    """Checkpoints of a run from log.txt (works while the run is still going): (samples, mem, new, others)."""
    rows = []
    for line in open(os.path.join(run, "log.txt")):
        if not line.startswith(f"{name} ep "):
            continue
        f = line.split()
        rows.append((int(f[2]), float(line.split(" mem ")[1].split()[0]),
                     float(line.split("->")[1].split()[0]), float(line.split(" others ")[1].split()[0])))
    return rows


def load_ltm(run, name):
    hist = json.load(open(os.path.join(run, f"{name}.json")))["result"]["history"]
    return [(h["samples"], h["new_train_recall"], h["new_recall"], h["others_acc"]) for h in hist]


def plot_repeats(args):
    """Wave 4: repeats per sample, memorisation and held-out recall vs samples, one figure per bias mode."""
    for mode in ("where", "what"):
        series = [("ltm200_lr0.001", "LTM (AdamW 1e-3)", "ltm")]
        series += [(f"w4_{mode}_a0.1_r{r}", f"STM {mode}, r={r}", "stm") for r in (1, 8, 16, 64, 128)]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
        partial = []
        for i, (name, label, kind) in enumerate(series):
            rows = load_ltm(args.run, name) if kind == "ltm" else load_log(args.run, name)
            if not rows:
                continue
            if kind == "stm" and not os.path.exists(os.path.join(args.run, f"{name}.json")):
                label += f" (running, {rows[-1][0]} samples)"
                partial.append(name)
            xs = [max(r[0], 5) for r in rows]
            for ax, k in zip(axes, (1, 2, 3)):
                ax.plot(xs, [r[k] for r in rows], color=SERIES[i], lw=2, ls="-" if kind == "ltm" else "--",
                        marker="o" if kind == "ltm" else "s", ms=5, mec=SURFACE, mew=1.5, label=label, zorder=3)
        titles = ("Memorised: recall on the 16 training nines", "Generalised: recall on held-out nines",
                  "Accuracy on digits 0-8")
        for ax, title in zip(axes, titles):
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log")
            ax.set_xlabel("samples presented (25% nines)")
            ax.set_title(title, color=INK, loc="left", fontsize=10.5)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
        axes[0].set_ylim(0, 1.02)
        axes[1].set_ylim(0, 1.02)
        axes[2].set_ylim(0.85, 1.0)
        axes[2].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
        fig.suptitle(f"Frozen gap snapshot: RL STM ({mode} bias, actor 0.1, actor sparse init) with r repeats per sample, "
                     f"1 other-digit episode between repeats, vs LTM", color=INK, x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()
        path = os.path.join(args.out, f"repeats_{mode}.png")
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        print(path, "partial:", partial)


WAVE5 = [  # (row title, [(file, label)]); all where, r64, interleave 1, actor 0.1, actor sparse init
    ("T = 4 steps, history 4", [
        ("w5_1_impr_T4H4", "improvement reward"),
        ("w5_2_acc_T4H4", "per-step accuracy reward"),
        ("w5_6_term_T4H4_lr0.03", "terminal reward, lr 0.03"),
        ("w5_7_impr_T4H4_lr0.03", "improvement reward, lr 0.03"),
    ]),
    ("T = 8 steps, step input", [
        ("w5_3_impr_T8H4_step", "improvement, history 4"),
        ("w5_4_impr_T8H8_step", "improvement, history 8"),
        ("w5_5_term_T8H8_step", "terminal, history 8"),
        ("w5_8_impr_T8H8_step_bonus0.01", "improvement, history 8, class bonus 1%"),
        ("w5_9_impr_T8H8_step_bonus0.05", "improvement, history 8, class bonus 5%"),
    ]),
]


def load_stm(run, name):
    hist = json.load(open(os.path.join(run, f"{name}.json")))["result"]["history"]
    return hist, [(h["samples"], h["new_train_recall"], h["new_recall"], h["others_acc"]) for h in hist]


def plot_wave5(args):
    """Wave 5: learning signal (reward, lr, episode length, history, step input, class bonus) vs samples, with the
    LTM and the wave-4 baseline (terminal reward, T 4) as references; last panel: accuracy after each step."""
    refs = [("ltm200_lr0.001", "LTM (AdamW 1e-3)", INK, "-", "o"),
            ("w4_where_a0.1_r64", "wave 4 baseline: terminal, T 4, lr 0.01", "#8f8d87", ":", "s")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6), facecolor=SURFACE)
    table = []
    for row, (row_title, runs) in enumerate(WAVE5):
        ax_row = axes[row]
        for name, label, color, ls, marker in refs:
            if name.startswith("ltm"):
                rows, steps = load_ltm(args.run, name), None
            else:
                hist, rows = load_stm(args.run, name)
                steps = hist[-1]["step_acc"]
            xs = [max(r[0], 5) for r in rows]
            for ax, k in zip(ax_row[:3], (1, 2, 3)):
                ax.plot(xs, [r[k] for r in rows], color=color, lw=2, ls=ls, marker=marker, ms=5, mec=SURFACE,
                        mew=1.5, label=label, zorder=3)
            if steps is not None:
                ax_row[3].plot(range(len(steps)), steps, color=color, lw=2, ls=ls, marker=marker, ms=5,
                               mec=SURFACE, mew=1.5, label=label, zorder=3)
            if row == 0:
                table.append((label, rows[-1][1], rows[-1][2], rows[-1][3], steps))
        for i, (name, label) in enumerate(runs):
            hist, rows = load_stm(args.run, name)
            xs = [max(r[0], 5) for r in rows]
            for ax, k in zip(ax_row[:3], (1, 2, 3)):
                ax.plot(xs, [r[k] for r in rows], color=SERIES[i], lw=2, ls="--", marker="s", ms=5, mec=SURFACE,
                        mew=1.5, label=label, zorder=3)
            steps = hist[-1]["step_acc"]
            ax_row[3].plot(range(len(steps)), steps, color=SERIES[i], lw=2, ls="--", marker="s", ms=5,
                           mec=SURFACE, mew=1.5, zorder=3)
            table.append((f"{label} ({row_title})", rows[-1][1], rows[-1][2], rows[-1][3], steps))
        titles = ("Memorised: recall on the 16 training nines", "Generalised: recall on held-out nines",
                  "Accuracy on digits 0-8", "All digits after each STM step, at 200 samples")
        for ax, title in zip(ax_row, titles):
            ax.set_facecolor(SURFACE)
            ax.set_title(title, color=INK, loc="left", fontsize=10.5)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for ax in ax_row[:3]:
            ax.set_xscale("log")
            ax.set_xlabel("samples presented (25% nines)")
        ax_row[0].set_ylim(0, 1.02)
        ax_row[1].set_ylim(0, 1.02)
        ax_row[2].set_ylim(0.85, 1.0)
        ax_row[3].set_ylim(0.87, 0.93)
        ax_row[3].set_xlabel("STM step (0 = unbiased LTM)")
        ax_row[3].set_xticks(range(9))
        ax_row[0].set_ylabel(row_title, color=INK, fontsize=11)
        ax_row[2].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle("Frozen gap snapshot, wave 5: RL STM learning signal (where bias, r 64, interleaved, actor 0.1, "
                 "SGD lr 0.01 unless noted) vs LTM, in samples", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    path = os.path.join(args.out, "wave5_learning_signal.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    with open(os.path.join(args.out, "wave5_learning_signal.md"), "w") as f:
        f.write("| run | memorised | held-out 9 | digits 0-8 | all digits, step 0 -> last |\n|---|---|---|---|---|\n")
        for label, m, n, o, steps in table:
            st = f"{steps[0]:.3f} -> {steps[-1]:.3f}" if steps else "n/a"
            f.write(f"| {label} | {m:.3f} | {n:.3f} | {o:.4f} | {st} |\n")
    print(path)


def load_ltm_any(run, name, max_samples=None):
    """LTM history as (samples, memorised or None, held-out recall, others) up to max_samples."""
    hist = json.load(open(os.path.join(run, f"{name}.json")))["result"]["history"]
    return [(h["samples"], h.get("new_train_recall"), h["new_recall"], h["others_acc"]) for h in hist
            if max_samples is None or h["samples"] <= max_samples]


def plot_curves(args, groups, refs, fname, title):
    """One row per group: memorised, held-out recall, others vs samples. refs: (name, label, colour, ls, kind, max
    samples) drawn in every row (kind "ltm" or "stm"); groups: (row title, [(name, label)]) in palette order."""
    fig, axes = plt.subplots(len(groups), 3, figsize=(16, 4.3 * len(groups)), facecolor=SURFACE, squeeze=False)
    for ax_row, (row_title, runs) in zip(axes, groups):
        row_max = max(load_stm(args.run, n)[1][-1][0] for n, _ in runs)
        # a ref with max_samples is a long-stream reference: only in rows that run longer than 200 samples, and
        # only from its first checkpoint after 0 (its checkpoints are sparse early on)
        series = [(n, l, c, ls, k, m) for n, l, c, ls, k, m in refs if m is None or row_max > 200]
        series += [(n, l, SERIES[i], "--", "stm", None) for i, (n, l) in enumerate(runs)]
        for name, label, color, ls, kind, max_samples in series:
            rows = load_ltm_any(args.run, name, max_samples) if kind == "ltm" else load_stm(args.run, name)[1]
            if max_samples is not None:
                rows = rows[1:]
            xs = [max(r[0], 2) for r in rows]
            for ax, k in zip(ax_row, (1, 2, 3)):
                pts = [(x, r[k]) for x, r in zip(xs, rows) if r[k] is not None]
                if pts:
                    ax.plot(*zip(*pts), color=color, lw=2, ls=ls, marker="o" if kind == "ltm" else "s", ms=4.5,
                            mec=SURFACE, mew=1.2, label=label, zorder=3)
        titles = ("Memorised: recall on the 16 training nines", "Generalised: recall on held-out nines",
                  "Accuracy on digits 0-8")
        for ax, t in zip(ax_row, titles):
            ax.set_facecolor(SURFACE)
            ax.set_title(t, color=INK, loc="left", fontsize=10.5)
            ax.grid(True, color=GRID, lw=0.8, zorder=0)
            ax.set_xscale("log")
            ax.set_xlabel("samples presented")
        ax_row[0].set_ylim(0, 1.02)
        ax_row[1].set_ylim(0, 1.02)
        ax_row[2].set_ylim(0.85, 1.0)
        ax_row[0].set_ylabel(row_title, color=INK, fontsize=11)
        ax_row[2].legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / (4.3 * len(groups))))
    path = os.path.join(args.out, fname)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def final_table(args, names, path, at=None):
    """Markdown table: memorised / held-out / others / |bias| / step acc at sample `at` (default: last)."""
    with open(path, "w") as f:
        f.write("| run | samples | memorised | held-out 9 | digits 0-8 | mean abs bias | all digits step 0 -> T |\n"
                "|---|---|---|---|---|---|---|\n")
        for name, label in names:
            hist = json.load(open(os.path.join(args.run, f"{name}.json")))["result"]["history"]
            h = hist[-1] if at is None else max((x for x in hist if x["samples"] <= at), key=lambda x: x["samples"])
            f.write(f"| {label} | {h['samples']} | {h['new_train_recall']:.3f} | {h['new_recall']:.3f} | "
                    f"{h['others_acc']:.4f} | {h['bias_abs']:.2f} | {h['step_acc'][0]:.3f} -> {h['step_acc'][-1]:.3f} |\n")
    print(open(path).read())


LTM_REFS = [("ltm200_lr0.001", "LTM (AdamW 1e-3)", INK, "-", "ltm", None),
            ("ltm_lr0.001", "LTM (AdamW 1e-3), longer stream", INK2, "-", "ltm", 1000)]


def plot_wave6(args):
    """Wave 6: per-step accuracy reward, T 4 H 4: lr x repeats, length, seeds, replay mixing."""
    base = ("w6_r64_lr0.01", "lr 0.01, r 64 (= wave 5 best)", "#8f8d87", ":", "stm", None)
    groups = [
        ("lr, 1000 samples", [("w6_long1000_lr0.01", "lr 0.01"), ("w6_long1000_lr0.03", "lr 0.03"),
                              ("w6_long1000_lr0.1", "lr 0.1")]),
        ("repeats, lr 0.01", [("w6_r16_lr0.01", "r 16"), ("w6_r128_lr0.01", "r 128"),
                              ("w6_actor0.3_lr0.01", "r 64, actor 0.3")]),
        ("lr 0.03: seeds, repeats", [("w6_r64_lr0.03", "r 64 seed 0"), ("w6_r64_lr0.03_s1", "r 64 seed 1"),
                                     ("w6_r64_lr0.03_s2", "r 64 seed 2"), ("w6_r16_lr0.03", "r 16"),
                                     ("w6_r128_lr0.03", "r 128")]),
        ("mixing, lr 0.03, r 64", [("w6_mix_i0_lr0.03", "interleave 0"), ("w6_mix_i4_lr0.03", "interleave 4 others"),
                                   ("w6_mix_i1stream_lr0.03", "interleave 1 replay mix"),
                                   ("w6_mix_i4stream_lr0.03", "interleave 4 replay mix"),
                                   ("w6_mix_nf0.1_lr0.03", "10% nines"), ("w6_mix_nf0.5_lr0.03", "50% nines")]),
    ]
    plot_curves(args, groups, [LTM_REFS[0], LTM_REFS[1], base], "wave6_per_step_accuracy.png",
                "Frozen gap snapshot, wave 6: RL STM, per-step accuracy reward, T 4, H 4 (where, actor 0.1, "
                "interleave 1 from other digits, 25% nines unless noted) vs LTM")
    names = [base[:2]] + [(n, f"{rt}: {l}") for rt, runs in groups for n, l in runs]
    names += [("w6_r16_lr0.1", "r 16 lr 0.1"), ("w6_r64_lr0.1", "r 64 lr 0.1"), ("w6_r128_lr0.1", "r 128 lr 0.1"),
              ("w6_actor0.3_lr0.03", "lr 0.03, actor 0.3")]
    final_table(args, names, os.path.join(args.out, "wave6_at200.md"), at=200)
    final_table(args, [r for r in groups[0][1]], os.path.join(args.out, "wave6_long.md"))


def plot_wave6b(args):
    """Wave 6b: reward scale p(y)^s, policy std, and mask keys that let the biased input choose the clique."""
    base = ("w6_r64_lr0.03", "base: lr 0.03, s 1, std 0.5, key first", "#8f8d87", ":", "stm", None)
    groups = [
        ("reward scale, exploration", [("w6b_s0.1", "s 0.1"), ("w6b_s0.3", "s 0.3"), ("w6b_std1.0", "std 1.0"),
                                       ("w6b_s0.1_std1.0", "s 0.1, std 1.0"), ("w6b_s0.1_lr0.01", "s 0.1, lr 0.01")]),
        ("clique key", [("w6b_key_current", "key current"), ("w6b_key_current_s0.1", "key current, s 0.1"),
                        ("w6b_key_history_s0.1", "key history, s 0.1")]),
    ]
    plot_curves(args, groups, [LTM_REFS[0], base], "wave6b_reward_key.png",
                "Frozen gap snapshot, wave 6b: RL STM, per-step accuracy reward p(y)^s, T 4, H 4 (where, r 64, "
                "interleave 1 from other digits, lr 0.03 unless noted) vs LTM")
    names = [base[:2], ("w6_r64_lr0.03_s1", "base seed 1"), ("w6_r64_lr0.03_s2", "base seed 2"),
             ("w6_r64_lr0.01", "lr 0.01, s 1")] + [r for _, runs in groups for r in runs]
    final_table(args, names, os.path.join(args.out, "wave6b_at200.md"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--figure", default="speed", choices=["speed", "repeats", "wave5", "wave6", "wave6b"])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    if args.figure == "repeats":
        plot_repeats(args)
        return
    if args.figure == "wave5":
        plot_wave5(args)
        return
    if args.figure == "wave6":
        plot_wave6(args)
        return
    if args.figure == "wave6b":
        plot_wave6b(args)
        return
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
