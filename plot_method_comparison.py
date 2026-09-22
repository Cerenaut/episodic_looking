"""
One summary figure comparing every model in the continual setting, to sit alongside the paper's
continual-learning metrics table (tab:cl-metrics).

The per-method figures from plot_comparison.py show four test-set curves per order (12 curves per
model), which is too much to compare models side by side. Here each run is collapsed to one curve by
aggregating over the four test sets at every epoch, and the three class orders (and any repeats) are
averaged, so each model is a single line:

    mean    mean accuracy over the four test sets (12, 3, 4, 5) -- the running version of the ACC
            column of the table: the last point of each curve IS that column.
    worst   accuracy on the worst of the four test sets. Averaging hides collapse (a model that has
            given up one class but gained another keeps its mean), so this panel is what separates
            "learned all four" from "learned the class it is currently trained on".

A third panel plots the stability-plasticity plane (LA against BWT) from the same runs, which is the
rest of the table in one view.

Each model's variant is a different set of runs (different code paths / checkpoints), so the sources
are listed in VARIANTS below rather than taken from the command line; they match the rows of
tab:cl-metrics and the per-variant directories in "Gideon Notes/figures/". --variants selects a
subset by key.

Epoch 0 of every curve is the pre-continual evaluation R[0, j] (the same baseline files the table's
BWT_0 and FWT are measured against), so the curves start where the model was before phase 1.

Usage (from the repo root, after the variants' runs are in place):
    python plot_method_comparison.py --out-dir runs_local/comparison_summary
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_comparison import (CHANCE_COLOUR, CHANCE_VALUE, FINE_NAMES, ORDERS, REGIME_CHANGE_COLOUR,
                             cl_metrics, collect, r_matrix)

# One entry per row of tab:cl-metrics: the runs it was computed from, and how it is drawn.
# glob patterns are relative to --runs-root; baselines holds the pre-continual evaluation R[0, j].
VARIANTS = [
    dict(key="stm_paper", label="CLS/STM (original code)", method="stm",
         globs=["runs/cifar_100/continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines",
         colour="tab:blue", linestyle=":", linewidth=2.0, family="CLS/STM"),
    dict(key="stm_fixed", label="CLS/STM (corrected RL)", method="stm",
         globs=["runs_ref_fixed/cifar_100/continual_[[]*[]]_500/**/results_continual.txt",
                "runs_ref_fixed_s2/cifar_100/continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines_ref_fixed_pooled",
         colour="tab:blue", linestyle="-", linewidth=3.0, family="CLS/STM"),
    dict(key="stm_pt40", label="CLS/STM (corrected RL, 40 pre-training epochs)", method="stm",
         globs=["runs_p0_pt40/cifar_100/continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines_pt40",
         colour="tab:blue", linestyle="--", linewidth=2.0, family="CLS/STM"),
    dict(key="stm_diff", label="CLS/STM (differentiable actor)", method="stm",
         globs=["runs_a1_diff_lr0.01/cifar_100/continual_[[]*[]]_500/**/results_continual.txt",
                "runs_a1_diff_s2/cifar_100/continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines_a1_pooled",
         colour="tab:cyan", linestyle="-", linewidth=2.0, family="CLS/STM"),
    dict(key="ltm", label="LTM-only fine-tuning", method="ltm",
         globs=["runs/cifar_100/continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines",
         colour="tab:red", linestyle="-", linewidth=3.0, family="Fine-tuned backbone"),
    dict(key="linear", label="Linear probe", method="linear",
         globs=["runs/cifar_100/head_linear_continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines",
         colour="dimgrey", linestyle="--", linewidth=2.0, family="Head on frozen LTM"),
    dict(key="ncm", label="NCM", method="ncm",
         globs=["runs/cifar_100/head_ncm_continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines",
         colour="tab:green", linestyle="-", linewidth=2.0, family="Head on frozen LTM"),
    dict(key="flymodel", label="FlyModel", method="flymodel",
         globs=["runs/cifar_100/head_flymodel_continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines",
         colour="tab:olive", linestyle="-.", linewidth=2.0, family="Head on frozen LTM"),
    dict(key="sdmlp", label="SDMLP", method="sdmlp",
         globs=["runs/cifar_100/head_sdmlp_continual_[[]*[]]_500/**/results_continual.txt"],
         baselines="runs_local/baselines",
         colour="tab:purple", linestyle="--", linewidth=2.0, family="Head on frozen LTM"),
]

PANELS = {
    "mean": ("Mean over the four test sets", "Mean accuracy"),
    "worst": ("Worst of the four test sets", "Worst-test-set accuracy"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=str, default=".")
    p.add_argument("--out-dir", type=str, default="runs_local/comparison_summary")
    p.add_argument("--variants", nargs="+", default=[v["key"] for v in VARIANTS],
                   help="Subset of VARIANTS keys to draw, in legend order.")
    p.add_argument("--panels", nargs="+", default=["mean", "worst"], choices=["mean", "worst", "plane"],
                   help="Panels of the figure: mean / worst accuracy curves, and the LA-BWT plane.")
    p.add_argument("--band", action="store_true",
                   help="Shade +/- 1 s.d. over the runs (orders x repeats) of each model.")
    p.add_argument("--style", choices=["paper", "screen"], default="paper",
                   help="paper: drawn at the size it is printed at (width=\\linewidth), so the text stays "
                        "readable after LaTeX scales it. screen: large panels for reading on their own.")
    p.add_argument("--width", type=float, default=None,
                   help="Total figure width in inches, overriding the style's. The paper style defaults to "
                        "PAPER_WIDTH (the NeurIPS \\linewidth), so \\includegraphics[width=\\linewidth] does not "
                        "scale the figure and the labels print at the size they are drawn at.")
    p.add_argument("--name", type=str, default="method_comparison", help="Output file stem.")
    return p.parse_args()


# \linewidth in the NeurIPS style, in inches: the paper figure is drawn this wide so that
# \includegraphics[width=\linewidth] scales it by 1 and the labels print at the size set here.
PAPER_WIDTH = 5.5

# Panel height and font sizes per style; the panel width follows from --width / PAPER_WIDTH for
# "paper" and is fixed for "screen".
STYLES = {
    "paper": dict(width=PAPER_WIDTH, height=2.05, font=6.5, title=7.0, line=0.6, legend=6.0, dpi=600,
                  max_xticks=5),
    "screen": dict(width=None, height=5.4, font=11.0, title=13.0, line=1.0, legend=10.0, dpi=200,
                   max_xticks=8),
}
SCREEN_PANEL_WIDTH = 7.2


def curves(variant: dict, runs_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Per-epoch aggregates over the four test sets, one row per epoch and one column per run.
    Returns (mean-over-test-sets, min-over-test-sets, per-run CL metrics). Epoch 0 is the
    pre-continual baseline R[0, j].
    """
    data = collect(runs_root, variant["globs"], variant["method"], ORDERS,
                   baselines_dir=os.path.join(runs_root, variant["baselines"]))
    mean_cols, worst_cols, metrics = {}, {}, []
    for order, pivots in data.items():
        for piv in pivots:
            cols = [c for c in FINE_NAMES if c in piv.columns]
            if len(cols) < len(FINE_NAMES):
                warnings.warn(f"{variant['key']}: {piv.attrs.get('path')} is missing test sets {set(FINE_NAMES) - set(cols)}")
            acc = piv[cols].astype(float)
            R = r_matrix(piv, order)
            base = R.loc[0, cols].astype(float)  # pre-continual evaluation, drawn at epoch 0
            name = f"{','.join(map(str, order))}#{piv.attrs.get('seed')}"
            # epochs are 1-based in the results files; re-index to 0..n so phases divide evenly
            mean_cols[name] = pd.Series([base.mean()] + list(acc.mean(axis=1)), index=range(len(acc) + 1))
            worst_cols[name] = pd.Series([base.min()] + list(acc.min(axis=1)), index=range(len(acc) + 1))
            metrics.append(cl_metrics(R, order))
    if not mean_cols:
        return pd.DataFrame(), pd.DataFrame(), []
    return pd.DataFrame(mean_cols), pd.DataFrame(worst_cols), metrics


def draw_curves(ax, series: dict, variants: list[dict], title: str, ylabel: str, band: bool, st: dict):
    n_epochs = max(len(s.index) - 1 for s in series.values())
    n_per_phase = n_epochs // len(ORDERS)
    for x in (n_per_phase, 2 * n_per_phase):
        ax.axvline(x, color=REGIME_CHANGE_COLOUR, linewidth=1.5 * st["line"], zorder=0)
    ax.axhline(CHANCE_VALUE, color=CHANCE_COLOUR, linestyle="-.", linewidth=1.2 * st["line"], zorder=0)
    for v in variants:
        df = series[v["key"]]
        if df.empty:
            continue
        m = df.mean(axis=1)
        ax.plot(m.index, m, label=v["label"], color=v["colour"], linestyle=v["linestyle"],
                linewidth=v["linewidth"] * st["line"], solid_capstyle="round")
        if band and df.shape[1] > 1:
            sd = df.std(axis=1, ddof=0)
            ax.fill_between(m.index, m - sd, m + sd, color=v["colour"], alpha=0.12, linewidth=0)
    ax.set_title(title, fontsize=st["title"], pad=6)
    ax.set_xlabel("Continual-learning epoch", fontsize=st["font"])
    ax.set_ylabel(ylabel, fontsize=st["font"])
    ax.set_xlim(0, n_epochs)
    ax.xaxis.set_major_locator(plt.MaxNLocator(st["max_xticks"], integer=True))
    ax.grid(True, alpha=0.25)
    # label the three phases along the top
    for i in range(len(ORDERS)):
        ax.annotate(f"phase {i + 1}", xy=((i + 0.5) * n_per_phase, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -1.3 * st["font"]), textcoords="offset points", ha="center",
                    fontsize=st["font"] - 1, color="grey")


def draw_plane(ax, metrics: dict, variants: list[dict], st: dict):
    """Stability-plasticity plane: learning accuracy (plasticity) against backward transfer (stability)."""
    for v in variants:
        rows = metrics[v["key"]]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        ax.scatter(df["LA"].mean(), df["BWT"].mean(), s=90 * st["line"] ** 2, color=v["colour"], label=v["label"],
                   marker="o" if v["family"] == "CLS/STM" else ("s" if v["family"] == "Head on frozen LTM" else "D"),
                   edgecolor="white", linewidth=1.2, zorder=3)
        if len(df) > 1:
            ax.errorbar(df["LA"].mean(), df["BWT"].mean(), xerr=df["LA"].std(ddof=0), yerr=df["BWT"].std(ddof=0),
                        color=v["colour"], alpha=0.5, linewidth=1.0 * st["line"], zorder=2)
    ax.axhline(0, color="black", linewidth=1.0 * st["line"], zorder=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(st["max_xticks"]))
    ax.set_xlabel("LA (plasticity)", fontsize=st["font"])
    ax.set_ylabel("BWT (stability)", fontsize=st["font"])
    ax.set_title("Stability-plasticity plane", fontsize=st["title"], pad=6)
    ax.grid(True, alpha=0.25)
    ax.margins(0.12)
    ax.annotate("", xy=(0.95, 0.93), xytext=(0.78, 0.76), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="grey", linewidth=st["line"]))
    ax.annotate("better", xy=(0.76, 0.79), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=st["font"], color="grey")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    variants = [v for k in args.variants for v in VARIANTS if v["key"] == k]
    if len(variants) != len(args.variants):
        missing = set(args.variants) - {v["key"] for v in VARIANTS}
        raise SystemExit(f"unknown variant(s): {sorted(missing)}")

    mean_s, worst_s, metrics, rows = {}, {}, {}, []
    for v in variants:
        m, w, met = curves(v, args.runs_root)
        mean_s[v["key"]], worst_s[v["key"]], metrics[v["key"]] = m, w, met
        if m.empty:
            warnings.warn(f"no runs found for variant {v['key']!r} (globs: {v['globs']}); it will be left out")
            continue
        print(f"{v['key']}: {m.shape[1]} run(s), {m.shape[0] - 1} epochs, final mean {m.iloc[-1].mean():.3f}")
        rows.append({"variant": v["key"], "label": v["label"], "n_runs": m.shape[1],
                     "final_mean": m.iloc[-1].mean(), "final_worst": w.iloc[-1].mean(),
                     "min_worst": w.mean(axis=1).min(),
                     **{k: float(np.mean([r[k] for r in met])) for k in ("ACC", "LA", "BWT", "BWT_0", "FWT")}})
    variants = [v for v in variants if not mean_s[v["key"]].empty]
    if not variants:
        raise SystemExit("no runs found for any variant")

    st = STYLES[args.style]
    n = len(args.panels)
    plt.rcParams.update({"xtick.labelsize": st["font"] - 1, "ytick.labelsize": st["font"] - 1})
    width = args.width or st["width"] or SCREEN_PANEL_WIDTH * n
    fig, axes = plt.subplots(1, n, figsize=(width, st["height"]), squeeze=False)
    for ax, panel in zip(axes[0], args.panels):
        if panel == "plane":
            draw_plane(ax, metrics, variants, st)
        else:
            series = mean_s if panel == "mean" else worst_s
            draw_curves(ax, series, variants, *PANELS[panel], band=args.band, st=st)
    lo = min(0.3, min(min(s[v["key"]].mean(axis=1).min() for v in variants)
                      for s in (mean_s, worst_s) if any(not s[v["key"]].empty for v in variants)))
    for ax, panel in zip(axes[0], args.panels):
        if panel != "plane":
            ax.set_ylim(max(0.0, lo - 0.05), 1.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False,
               fontsize=st["legend"], bbox_to_anchor=(0.5, -0.02), handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    path = os.path.join(args.out_dir, f"{args.name}.png")
    fig.savefig(path, dpi=st["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    csv_path = os.path.join(args.out_dir, f"{args.name}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
