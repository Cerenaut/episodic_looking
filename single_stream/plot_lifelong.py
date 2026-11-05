"""
Lifelong sweep 1 (Dave 5 Nov 2026): LTM + replay + sleep over 4 tasks (1 epoch each, instant switches), two
families analysed separately (dih: dihedral image transforms; lp: label permutations), against cluster size c and the
sleep schedule (period P, steps K, rate r = K / P). Figures and tables for the report; every figure shows the two
families side by side.

Metrics per post-switch task (tasks 2-4; task 1 is learning from scratch):
  Plasticity (final)       accuracy on the task at its end;
  Anytime                  mean accuracy on the task over its uniform grid of checkpoints;
  Resilience               recovery after the switch: accuracy at 64 / 256 / 1,024 / 4,096 exposures, and the
                           exposures to reach 90% of the task's final accuracy;
  Zero-shot                accuracy on the new task at the switch, before its first sample;
  Retention                accuracy on task 1 (the original) at the end of the run, and forgetting (accuracy on each
                           earlier task at its end minus at the end of the run);
  Protection               the worst class over the task's grid (mean of the worst class at each checkpoint);
  Generalisation, Memorisation, Protection at the few-shot triggers (a class's n-th exposure within the task).

    python -W ignore -m single_stream.plot_lifelong --run runs_ss/mnist_clustered_single_stream/sweep1 --out <dir>
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from single_stream.plot_stm_frozen import INK, SERIES, SURFACE
from single_stream.plot_wave8 import GREY, save, style

FAMILIES = [("dih", "Dihedral transforms (id, rot90, hflip, rot180)"),
            ("lp", "Label permutations (id, lperm1, lperm2, lperm3)")]
CS = (1, 32, 128, 256, 512)
RATES = (1 / 64, 1 / 16, 1 / 4, 1)
RATE_LABEL = {1 / 64: "1/64", 1 / 16: "1/16", 1 / 4: "1/4", 1: "1"}
PK = {1 / 64: [(64, 1)], 1 / 16: [(16, 1), (64, 4)], 1 / 4: [(16, 4), (64, 16)], 1: [(16, 16), (64, 64)]}
# one colour per sleep rate (sequential: more sleep = darker), one per cluster size (categorical)
RATE_COLOUR = {1 / 64: "#9ec5f0", 1 / 16: "#5a9be0", 1 / 4: "#2a78d6", 1: "#0d3d75"}
C_COLOUR = dict(zip(CS, [INK, SERIES[0], SERIES[1], SERIES[2], SERIES[4]]))
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9ec5f0", "#2a78d6", "#0d3d75"])
RUN = None
RX = re.compile(r"(dih|lp)_c(\d+)_P(\d+)_K(\d+)_s(\d+)$")


def load():
    """{(fam, c, P, K): [run dicts over seeds]}; run = summary, checkpoints, fewshot."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(RUN, "*"))):
        m = RX.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "summary.json")):
            continue
        fam, c, P, K, s = m.group(1), *map(int, m.groups()[1:])
        r = {"summary": json.load(open(os.path.join(d, "summary.json"))),
             "ckpt": [json.loads(l) for l in open(os.path.join(d, "checkpoints.jsonl"))],
             "fewshot": [json.loads(l) for l in open(os.path.join(d, "fewshot.jsonl"))], "seed": s}
        runs.setdefault((fam, c, P, K), []).append(r)
    return runs


def rate(P, K):
    return K / P


def task_metrics(run):
    """Per-run scalar metrics, averaged over the post-switch tasks (2-4), plus task 1's."""
    per = run["summary"]["per_task"]
    names = run["summary"]["tasks"]
    post = per[1:]
    out = {
        "final": np.mean([p["final"] for p in post]), "anytime": np.mean([p["anytime"] for p in post]),
        "zero_shot": np.mean([p["zero_shot"] for p in post]),
        "worst": np.mean([p["worst_group_anytime"] for p in post]),
        "task1_final": per[0]["final"], "task1_anytime": per[0]["anytime"],
        "retention_task1": per[-1]["task_acc_end"][names[0]],
        "forgetting": np.mean([per[i]["final"] - per[-1]["task_acc_end"][names[i]] for i in range(len(per) - 1)]),
    }
    for x in ("64", "256", "1024", "4096"):
        out[f"rec_{x}"] = np.mean([p["recovery"][x] for p in post])
    # exposures to reach 90% of the task's final accuracy, from the task's checkpoints
    ts = []
    for k in range(1, len(names)):
        recs = [r for r in run["ckpt"] if r["task"] == k]
        goal = 0.9 * per[k]["final"]
        ts.append(next((r["t_task"] for r in recs if r["acc"] >= goal), recs[-1]["t_task"]))
    # geometric mean over tasks; undefined when a task is never learned (final below 0.3: chance is 0.1)
    learned = all(per[k]["final"] >= 0.3 for k in range(1, len(names)))
    out["t90"] = float(np.exp(np.mean(np.log(np.maximum(ts, 1))))) if learned else np.nan
    return out


def metric(runs, key, fam, c, P, K):
    rs = runs.get((fam, c, P, K), [])
    return float(np.mean([task_metrics(r)[key] for r in rs])) if rs else np.nan


def metric_rate(runs, key, fam, c, r, P=None):
    """At sleep rate r: P = 16 where it exists (else 64), or the given P."""
    cands = [(p, k) for p, k in PK[r] if P is None or p == P]
    for p, k in cands:
        v = metric(runs, key, fam, c, p, k)
        if not np.isnan(v):
            return v
    return np.nan


METRICS = [("anytime", "Anytime accuracy, tasks 2-4", (0.3, 1.0)),
           ("final", "Final accuracy (plasticity), tasks 2-4", (0.3, 1.0)),
           ("rec_1024", "Resilience: acc. 1,024 after switch", (0.0, 1.0)),
           ("t90", "Resilience: exposures to 90% of final (lower better)", None),
           ("worst", "Protection: worst class (anytime), tasks 2-4", (0.0, 1.0)),
           ("retention_task1", "Retention: task 1 accuracy at the end", (0.0, 1.0)),
           ("forgetting", "Forgetting of earlier tasks (lower = better)", None),
           ("zero_shot", "Zero-shot accuracy at the switch", (0.0, 1.0))]


def fig_heatmaps(args, runs):
    """Bivariate: c x sleep rate, one heatmap per metric and family."""
    for chunk, name in ((METRICS[:4], "lifelong_heat_a.png"), (METRICS[4:], "lifelong_heat_b.png")):
        fig, axes = plt.subplots(len(chunk), 2, figsize=(11, 3.1 * len(chunk)), facecolor=SURFACE)
        for row, (key, title, lim) in zip(axes, chunk):
            vals = {f: np.array([[metric_rate(runs, key, f, c, r) for r in RATES] for c in CS]) for f, _ in FAMILIES}
            allv = np.concatenate([v.ravel() for v in vals.values()])
            allv = allv[~np.isnan(allv)]
            if not len(allv):
                continue
            vmin, vmax = (lim if lim else (allv.min(), allv.max()))
            if key == "t90":
                vmin, vmax = allv.min(), allv.max()
            for ax, (f, flabel) in zip(row, FAMILIES):
                M = vals[f]
                cmap = SEQ.reversed() if key in ("t90", "forgetting") else SEQ
                ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                for i in range(len(CS)):
                    for j in range(len(RATES)):
                        v = M[i, j]
                        if np.isnan(v):
                            continue
                        txt = f"{v:,.0f}" if key == "t90" else f"{v:.2f}"
                        dark = (v - vmin) / max(vmax - vmin, 1e-9)
                        dark = 1 - dark if key in ("t90", "forgetting") else dark
                        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                                color=SURFACE if dark > 0.55 else INK)
                ax.set_xticks(range(len(RATES)))
                ax.set_xticklabels([RATE_LABEL[r] for r in RATES])
                ax.set_yticks(range(len(CS)))
                ax.set_yticklabels([str(c) for c in CS])
                ax.set_xlabel("sleep rate r = steps per exposure")
                ax.set_ylabel("cluster size c")
                ax.set_title(f"{flabel.split(' (')[0].replace(' transforms', '').replace(' permutations', ' perm.')}: {title}", color=INK, loc="left", fontsize=9.5)
        fig.suptitle("Cluster size x sleep rate (mean of seeds; P = 16 where it exists, else 64)", color=INK, x=0.01,
                     ha="left", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save(fig, args, name)


def fig_univariate(args, runs):
    """Each metric against c (lines: sleep rate) and against sleep rate (lines: c), families side by side."""
    keys = [m for m in METRICS if m[0] in ("anytime", "rec_1024", "worst", "retention_task1")]
    fig, axes = plt.subplots(len(keys), 4, figsize=(17, 3.3 * len(keys)), facecolor=SURFACE)
    for row, (key, title, lim) in zip(axes, keys):
        for k, (f, flabel) in enumerate(FAMILIES):
            ax = row[k]
            for r in RATES:
                y = [metric_rate(runs, key, f, c, r) for c in CS]
                ax.plot(range(len(CS)), y, color=RATE_COLOUR[r], lw=2, marker="o", ms=4, label=f"r {RATE_LABEL[r]}")
            ax.set_xticks(range(len(CS)))
            ax.set_xticklabels([str(c) for c in CS])
            style(ax, f"{f}: {title}", xlabel="cluster size c")
            ax = row[2 + k]
            for c in CS:
                y = [metric_rate(runs, key, f, c, r) for r in RATES]
                ax.plot(range(len(RATES)), y, color=C_COLOUR[c], lw=2, marker="o", ms=4, label=f"c {c}")
            ax.set_xticks(range(len(RATES)))
            ax.set_xticklabels([RATE_LABEL[r] for r in RATES])
            style(ax, f"{f}: {title}", xlabel="sleep rate r")
            if lim:
                row[k].set_ylim(*lim)
                row[2 + k].set_ylim(*lim)
        for ax in row:
            ax.title.set_fontsize(8.5)
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    axes[0, 2].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("One variable at a time: against cluster size (left pair) and against sleep rate (right pair); "
                 "dih = dihedral, lp = label permutation", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, args, "lifelong_univariate.png")


def mean_curve(rs, key="acc"):
    """Seed-mean curve over the whole run: global t and the current task's accuracy (or task-1 accuracy)."""
    T = None
    ys = []
    for r in rs:
        t = np.array([x["t"] for x in r["ckpt"]])
        y = np.array([x["acc"] if key == "acc" else x["task_acc"][r["summary"]["tasks"][0]] for x in r["ckpt"]])
        T = t if T is None else T
        ys.append(np.interp(T, t, y))
    return T, np.mean(ys, 0)


def fig_timelines(args, runs):
    """Higher dimensions as small multiples: rows c, columns family; lines sleep rate; the whole 4-task run."""
    fig, axes = plt.subplots(len(CS), 2, figsize=(15, 2.5 * len(CS)), facecolor=SURFACE, sharex=True, sharey=True)
    for i, c in enumerate(CS):
        for j, (f, flabel) in enumerate(FAMILIES):
            ax = axes[i, j]
            for r in RATES:
                P, K = next(((p, k) for p, k in PK[r] if (f, c, p, k) in runs), (None, None))
                if P is None:
                    continue
                T, y = mean_curve(runs[(f, c, P, K)])
                ax.plot(T / 1000, y, color=RATE_COLOUR[r], lw=1.6, label=f"r {RATE_LABEL[r]}")
                if r == 1 / 16:
                    T, y1 = mean_curve(runs[(f, c, P, K)], "task1")
                    ax.plot(T / 1000, y1, color=GREY, lw=1, ls=":", label="task 1 accuracy (r 1/16)")
            for k in range(1, 4):
                ax.axvline(54 * k, color=GREY, lw=0.8, ls="--")
            style(ax, f"{flabel.split(' (')[0]}, c = {c}", xlabel="exposures (thousands); dashed: task switches"
                  if i == len(CS) - 1 else "")
            ax.set_ylim(0, 1)
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK, ncol=2)
    fig.suptitle("The whole run: accuracy on the current task (solid) and on task 1 (dotted), mean of seeds",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, args, "lifelong_timelines.png")


def recovery_curve(rs, points):
    """Accuracy against exposures since the switch (log points), mean over tasks 2-4 and seeds."""
    out = []
    for r in rs:
        per_task = []
        for k in range(1, 4):
            recs = [x for x in r["ckpt"] if x["task"] == k]
            t = np.array([x["t_task"] for x in recs])
            y = np.array([x["acc"] for x in recs])
            per_task.append(np.interp(points, t, y))
        out.append(np.mean(per_task, 0))
    return np.mean(out, 0)


def fig_recovery(args, runs):
    points = np.unique(np.concatenate([[0], np.geomspace(1, 54000, 60)]))
    fig, axes = plt.subplots(2, len(CS), figsize=(18, 7.4), facecolor=SURFACE, sharex=True, sharey=True)
    for j, (f, flabel) in enumerate(FAMILIES):
        for i, c in enumerate(CS):
            ax = axes[j, i]
            for r in RATES:
                P, K = next(((p, k) for p, k in PK[r] if (f, c, p, k) in runs), (None, None))
                if P is None:
                    continue
                ax.plot(points + 1, recovery_curve(runs[(f, c, P, K)], points), color=RATE_COLOUR[r], lw=1.8,
                        label=f"r {RATE_LABEL[r]}")
            ax.set_xscale("log")
            style(ax, f"{f}, c = {c}", xlabel="exposures since the switch + 1" if j == 1 else "")
            ax.set_ylim(0, 1)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Resilience: recovery after a switch (accuracy on the new task, mean of tasks 2-4 and seeds)",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "lifelong_recovery.png")


def fig_fewshot(args, runs):
    """Few-shot triggers within tasks 2-4: Generalisation (recall), Memorisation, Protection against n."""
    ns = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    keys = [("recall", "Generalisation (held-out recall of the class)"),
            ("memorisation", "Memorisation (its training samples so far)"),
            ("protection", "Protection (the other classes)")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6), facecolor=SURFACE, sharex=True, sharey=True)
    for j, (f, flabel) in enumerate(FAMILIES):
        for i, (key, title) in enumerate(keys):
            ax = axes[j, i]
            for c in (1, 128, 512):
                for r, ls in ((1 / 16, "--"), (1, "-")):
                    P, K = next(((p, k) for p, k in PK[r] if (f, c, p, k) in runs), (None, None))
                    if P is None:
                        continue
                    y = []
                    for n in ns:
                        v = [x[key] for run in runs[(f, c, P, K)] for x in run["fewshot"]
                             if x["task"] >= 1 and x["threshold"] == n]
                        y.append(np.mean(v) if v else np.nan)
                    ax.plot(ns, y, color=C_COLOUR[c], ls=ls, lw=1.8, marker="o", ms=3,
                            label=f"c {c}, r {RATE_LABEL[r]}")
            ax.set_xscale("log", base=2)
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns])
            style(ax, f"{f}: {title}", xlabel="the class's n-th exposure in the task" if j == 1 else "")
            ax.set_ylim(0, 1)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", frameon=False, fontsize=9, labelcolor=INK, ncol=len(l))
    fig.suptitle("Few-shot within tasks 2-4 (each class's n-th exposure after the switch; mean of classes, tasks, "
                 "seeds)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(fig, args, "lifelong_fewshot.png")


def fig_period(args, runs):
    """Sleep period at a fixed rate: P 16 against P 64 (anytime and recovery at 256)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax, key, title in zip(axes, ("anytime", "rec_256"), ("Anytime accuracy, tasks 2-4",
                                                               "Accuracy 256 exposures after the switch")):
        for (f, _), marker in zip(FAMILIES, ("o", "s")):
            for c in CS:
                for r in (1 / 16, 1 / 4, 1):
                    a, b = metric_rate(runs, key, f, c, r, P=16), metric_rate(runs, key, f, c, r, P=64)
                    ax.scatter(a, b, color=C_COLOUR[c], marker=marker, s=40, edgecolor=SURFACE, zorder=3)
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", zorder=1)
        style(ax, title, xlabel="sleep every 16 exposures")
        ax.set_ylabel("sleep every 64 exposures (same rate)")
    h = [plt.Line2D([], [], ls="", marker="o", color=C_COLOUR[c], label=f"c {c}") for c in CS]
    h += [plt.Line2D([], [], ls="", marker=m, color=GREY, label=f) for f, m in (("dih", "o"), ("lp", "s"))]
    axes[0].legend(handles=h, loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle("Does the sleep period matter at a fixed sleep rate? (points on the diagonal: no)", color=INK,
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, args, "lifelong_period.png")


def fig_family(args, runs):
    """Which family is harder? Each configuration (c, P, K) as a point: dihedral against label permutation."""
    panels = [("zero_shot", "Zero-shot at the switch"), ("rec_256", "Accuracy 256 after the switch"),
              ("rec_1024", "Accuracy 1,024 after the switch"), ("anytime", "Anytime accuracy"),
              ("worst", "Protection: worst class"), ("final", "Final accuracy"),
              ("retention_task1", "Retention: task 1 at the end")]
    markers = {1 / 64: "v", 1 / 16: "s", 1 / 4: "D", 1: "o"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.ravel(), panels):
        for c in CS:
            for r in RATES:
                for P, K in PK[r]:
                    a, b = metric(runs, key, "dih", c, P, K), metric(runs, key, "lp", c, P, K)
                    ax.scatter(a, b, color=C_COLOUR[c], marker=markers[r], s=36, edgecolor=SURFACE, zorder=3)
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style(ax, f"{title} (tasks 2-4)" if key != "retention_task1" else title, xlabel="dihedral")
        ax.set_ylabel("label permutation")
    ax = axes.ravel()[-1]
    ax.axis("off")
    h = [plt.Line2D([], [], ls="", marker="o", color=C_COLOUR[c], label=f"c {c}") for c in CS]
    h += [plt.Line2D([], [], ls="", marker=m, color=GREY, label=f"r {RATE_LABEL[r]}") for r, m in markers.items()]
    ax.legend(handles=h, loc="center", frameon=False, fontsize=10, labelcolor=INK, ncol=2)
    fig.suptitle("Dihedral against label permutation, configuration by configuration (above the diagonal: label "
                 "permutation is easier)", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args, "lifelong_family.png")


def fig_seeds(args, runs):
    fig, ax = plt.subplots(figsize=(5.6, 5), facecolor=SURFACE)
    for (f, c, P, K), rs in runs.items():
        if len(rs) >= 2:
            a, b = (task_metrics(r)["anytime"] for r in sorted(rs, key=lambda r: r["seed"])[:2])
            ax.scatter(a, b, color=C_COLOUR[c], marker="o" if f == "dih" else "s", s=30, edgecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    style(ax, "Seed 0 against seed 1: anytime accuracy, tasks 2-4", xlabel="seed 0")
    ax.set_ylabel("seed 1")
    save(fig, args, "lifelong_seeds.png")


def fig_probe(args, probe):
    if not probe:
        return
    T = probe["tasks"]
    names = list(T)
    keys = [("zero_shot", "zero-shot (LTM trained on the original)"), ("shared_bias", "best single shared bias"),
            ("image_bias", "best bias per image (label-supervised)"), ("head_probe", "retrained classifier head")]
    fig, ax = plt.subplots(figsize=(14, 4.6), facecolor=SURFACE)
    w = 0.2
    for i, (k, label) in enumerate(keys):
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, [T[n][k] for n in names], width=w - 0.02,
               color=[INK, SERIES[0], SERIES[1], SERIES[3]][i], label=label)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    style(ax, "Each task seen from an LTM trained on the original (validation accuracy)", xlabel="")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, args, "lifelong_probe.png")


def table(args, runs, probe):
    L = ["## Sweep 1: per configuration (mean of seeds; tasks 2-4 unless stated)", "",
         "| family | c | P | K | r | seeds | anytime | final | zero-shot | acc @64 / 256 / 1024 / 4096 | t90 | worst class | "
         "task 1 final / anytime | task 1 at end | forgetting |", "|" + "---|" * 15]
    for (f, c, P, K), rs in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1], rate(*kv[0][2:]), kv[0][2])):
        m = {k: np.mean([task_metrics(r)[k] for r in rs]) for k in task_metrics(rs[0])}
        L.append(f"| {f} | {c} | {P} | {K} | {RATE_LABEL.get(rate(P, K), rate(P, K))} | {len(rs)} | {m['anytime']:.3f} | "
                 f"{m['final']:.3f} | {m['zero_shot']:.3f} | {m['rec_64']:.2f} / {m['rec_256']:.2f} / "
                 f"{m['rec_1024']:.2f} / {m['rec_4096']:.2f} | {m['t90']:,.0f} | {m['worst']:.3f} | "
                 f"{m['task1_final']:.3f} / {m['task1_anytime']:.3f} | {m['retention_task1']:.3f} | {m['forgetting']:.3f} |")
    if probe:
        L += ["", "## Transform probe (LTM trained iid on the original; validation accuracy)", "",
              "| task | zero-shot | shared bias | per-image bias | head probe |", "|---|---|---|---|---|"]
        for n, v in probe["tasks"].items():
            L.append(f"| {n} | {v['zero_shot']:.3f} | {v['shared_bias']:.3f} | {v['image_bias']:.3f} | "
                     f"{v['head_probe']:.3f} |")
    path = os.path.join(args.out, "lifelong_tables.md")
    open(path, "w").write("\n".join(L) + "\n")
    print(path)


def main():
    global RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RUN = args.run
    os.makedirs(args.out, exist_ok=True)
    probe = json.load(open(args.probe)) if args.probe and os.path.exists(args.probe) else None
    runs = load()
    print(f"{len(runs)} configurations, {sum(len(v) for v in runs.values())} runs")
    fig_probe(args, probe)
    fig_heatmaps(args, runs)
    fig_univariate(args, runs)
    fig_timelines(args, runs)
    fig_recovery(args, runs)
    fig_fewshot(args, runs)
    fig_period(args, runs)
    fig_family(args, runs)
    fig_seeds(args, runs)
    table(args, runs, probe)


if __name__ == "__main__":
    main()
