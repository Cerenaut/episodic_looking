"""
Early-learning delay of the RL STM from the per-sample diagnostics (stm_frozen rl --diag, <name>_diag.jsonl).

For each presentation of a few-shot image (the new digit): the STM's p(label) under the policy mean before its block
of repeats (pre) and after it (post), the spread of the episode return over the block (the reward signal exploration
produced), the critic's baseline error, gradient norms. From these:

  within-block gain   post - pre: what one presentation (r repeats) teaches
  retention           pre at the next presentation of the same image - post at the previous one (negative: forgotten
                      in between, by updates on other samples)
  flat reward         return spread vs pre: images the policy cannot yet move give no learning signal

  python -m single_stream.analyse_stm_diag --run runs_ss/stm_frozen_20261101/snap9 --names w6_long1000_lr0.01 ... --out <dir>
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(run, name):
    return [json.loads(line) for line in open(os.path.join(run, f"{name}_diag.jsonl"))]


def reward_scale(run, name):
    return json.load(open(os.path.join(run, f"{name}.json")))["args"].get("reward_scale", 1.0)


def presentations(diag, max_sample=None, scale=1.0):
    """Few-shot presentations: list of dicts with sample, index, k (earlier presentations), pre, post, prev_post,
    return_std, baseline_error, grad_*. Scores are logged as p(y)^scale (the reward's score); converted back to p(y)
    here so runs with different reward scales compare (return_std stays in each run's reward units)."""
    def p(score):
        return score ** (1.0 / scale)
    last, out = {}, []
    for d in diag:
        if max_sample is not None and d["sample"] > max_sample:
            break
        if not d["new"]:
            continue
        t = d["target"]
        if "p_T" in d["pre"]:  # p(y) logged directly (from wave 8; needed for non-probability rewards)
            p0, pre, post = d["pre"]["p_0"], d["pre"]["p_T"], d["post"]["p_T"]
        else:
            p0, pre, post = p(d["pre"]["score_0"]), p(d["pre"]["score_T"]), p(d["post"]["score_T"])
        out.append({"sample": d["sample"], "index": d["index"], "k": d["times_presented"],
                    "p0": p0, "pre": pre, "post": post,
                    "prev_post": last.get(d["index"]), "return_std": t["return_std"],
                    "baseline_error": t["baseline_error"], "grad_actor_out": t["grad_actor_out"]})
        last[d["index"]] = post
    return out


def summary(pres):
    gain = np.array([p["post"] - p["pre"] for p in pres])
    ret = np.array([p["pre"] - p["prev_post"] for p in pres if p["prev_post"] is not None])
    pre = np.array([p["pre"] for p in pres])
    std = np.array([p["return_std"] for p in pres])
    low, high = pre < 0.1, pre >= 0.3
    return {"presentations": len(pres), "gain": float(gain.mean()),
            "gain_first": float(np.mean([p["post"] - p["pre"] for p in pres if p["k"] == 0])),
            "retention": float(ret.mean()) if len(ret) else float("nan"),
            "forgot_0.2": float((ret < -0.2).mean()) if len(ret) else float("nan"),
            "return_std_low": float(std[low].mean()) if low.any() else float("nan"),
            "return_std_high": float(std[high].mean()) if high.any() else float("nan"),
            "gain_low": float(gain[low].mean()) if low.any() else float("nan"),
            "baseline_error_abs": float(np.mean([abs(p["baseline_error"]) for p in pres]))}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)


def plot(run, names, labels, out, max_sample):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    data = {n: presentations(load(run, n), max_sample, reward_scale(run, n)) for n in names}

    # Figure 1: trajectories of each few-shot image (first run), flat reward, within-block gain
    pres = data[names[0]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), facecolor=SURFACE)
    images = sorted({p["index"] for p in pres}, key=lambda i: next(p["p0"] for p in pres if p["index"] == i))
    cmap = plt.get_cmap("viridis")
    for n, i in enumerate(images):
        ps = [p for p in pres if p["index"] == i]
        xs, ys = [], []
        for p in ps:
            xs += [p["sample"], p["sample"]]
            ys += [p["pre"], p["post"]]
        axes[0].plot(xs, ys, color=cmap(n / max(1, len(images) - 1)), lw=1.4, marker="o", ms=3, zorder=3)
    style(axes[0], "p(9) of each training nine, before / after each presentation")
    axes[0].set_xlabel("samples presented (colour: the image's unbiased p(9), purple low to yellow high)")
    axes[0].set_ylabel("p(9) after T steps, policy mean")
    for j, n in enumerate(names):
        ps = data[n]
        axes[1].scatter([p["pre"] for p in ps], [p["return_std"] for p in ps], s=16, color=SERIES[j],
                        edgecolor=SURFACE, lw=0.5, label=labels[j], zorder=3)
        axes[2].scatter([p["pre"] for p in ps], [p["post"] - p["pre"] for p in ps], s=16, color=SERIES[j],
                        edgecolor=SURFACE, lw=0.5, label=labels[j], zorder=3)
    style(axes[1], "Reward signal: return spread over the repeats")
    axes[1].set_xlabel("p(9) before the presentation")
    axes[1].set_ylabel("std of the episode return over r repeats")
    style(axes[2], "What one presentation (r repeats) teaches")
    axes[2].set_xlabel("p(9) before the presentation")
    axes[2].set_ylabel("p(9) after - before")
    axes[2].axhline(0, color=INK2, lw=0.8)
    axes[2].legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle(f"RL STM delay diagnostics: few-shot nines ({labels[0]} on the left)", color=INK, x=0.01,
                 ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "delay_presentations.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: retention between presentations of the same image
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 5), facecolor=SURFACE)
    for j, n in enumerate(names):
        ps = [p for p in data[n] if p["prev_post"] is not None]
        ax.scatter([p["prev_post"] for p in ps], [p["pre"] for p in ps], s=16, color=SERIES[j], edgecolor=SURFACE,
                   lw=0.5, label=labels[j], zorder=3)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=":")
    style(ax, "Retention between presentations")
    ax.set_xlabel("p(9) after the previous presentation")
    ax.set_ylabel("p(9) at the next presentation")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "delay_retention.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(out, "delay_summary.md"), "w") as f:
        keys = list(summary(data[names[0]]).keys())
        f.write(f"Few-shot presentations up to sample {max_sample}.\n\n| run | " + " | ".join(keys) + " |\n")
        f.write("|---|" + "---|" * len(keys) + "\n")
        for n, label in zip(names, labels):
            s = summary(data[n])
            f.write(f"| {label} | " + " | ".join(f"{s[k]:.3f}" if isinstance(s[k], float) else str(s[k])
                                                 for k in keys) + " |\n")
    print(open(os.path.join(out, "delay_summary.md")).read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--max-sample", type=int, default=200, help="compare runs over the same samples")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plot(args.run, args.names, args.labels or args.names, args.out, args.max_sample)


if __name__ == "__main__":
    main()
