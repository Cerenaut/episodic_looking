"""
Compare the RL-trained STM, the fine-tuned LTM and the four frozen-encoder baseline heads
(cifar_main_head_baselines.py) in the continual and streaming settings.

Inputs are CifarResults files (results_<suffix>.txt) found by glob. Default patterns (relative to
--runs-root, recursive), one list per method; override any of them with --glob METHOD=PATTERN
(repeatable; the pattern replaces the defaults for that method):

    stm       runs*/cifar_100/continual_[*]_500/**/results_continual.txt     (cifar_main_stm_training.py)
              runs*/cifar_100/stm*continual_[*]_500/**/results_*continual.txt
    ltm       runs*/cifar_100/ltm*continual_[*]_500/**/results_*continual.txt (cifar_main_ltm_fine_tuning.py)
    linear    runs*/cifar_100/head_linear_continual_[*]_500/**/results_continual.txt
    ncm       runs*/cifar_100/head_ncm_continual_[*]_500/**/results_continual.txt
    flymodel  runs*/cifar_100/head_flymodel_continual_[*]_500/**/results_continual.txt
    sdmlp     runs*/cifar_100/head_sdmlp_continual_[*]_500/**/results_continual.txt

  streaming (batch size 1, single fine class):
    stm       runs*/cifar_100/few-shot_[?]_500/**/results_few-shot.txt   and  streaming_[?]_500/**/results_streaming.txt
    ltm       runs*/cifar_100/ltm*few-shot_[?]_500/**/results_*few-shot.txt  and  ltm*streaming_[?]_500/**/results_*streaming.txt
    <head>    runs*/cifar_100/head_<head>_streaming_[?]_500/**/results_streaming.txt

NOTE: the paper's two reference scripts both name their experiment "continual_[3, 4, 5]_500". STM and
LTM files are told apart by the path when it contains "stm" or "ltm" (e.g. runs_stm/, runs_ltm/, or an
"ltm-continual" prefix as in the paper's notebook); otherwise by content: the LTM script writes the whole
order "[3, 4, 5]" in its training lines while the STM script writes the current class "[3]" only.

The order (fine-class sequence) is parsed from the "[3, 4, 5]" in the path, the seed from a
"seed_<n>" path component if present (otherwise each file is a separate seed).

Outputs (in --out-dir, default runs_local/comparison/):
    <method>_continual.png       3 panels (orders), lines for 12/3/4/5, mean +/- std over seeds
    <method>_streaming.png       3 panels (fine class 3, 4, 5)
    summary_continual.csv / .md  per method x order (+ a per-method row pooled over orders): final acc. on
                                 12/3/4/5, mean final, forgetting, and the standard continual-learning metrics
                                 ACC, ACC_new, BWT, BWT_0, FWT, forgetting (end-of-phase / per-epoch max), LA
                                 (definitions: see cl_metrics()).
    R_matrices.csv               the accuracy matrix R[phase, test set] per method/order/seed (continual)
    summary_streaming.csv / .md

Continual-learning metrics need R[0, j], the accuracy on each test set BEFORE the continual phase:
    heads     the last evaluate rows of results_pretrain.txt next to results_continual.txt (the head scripts
              evaluate all four test sets after every pretraining epoch on fine-classes 1,2)
    ltm       --baselines-dir/ltm_pretrained_baseline.txt, written by eval_pretrained_baseline.py (the LTM
              script only evaluates after training epochs)
    stm       --baselines-dir/stm_pretrained_baseline.txt if present; otherwise the STM's epoch-1 evaluation is
              used as a proxy and FWT / BWT_0 are footnoted in the markdown summary.
Paths containing "quarantine" are ignored.
"""
import argparse
import glob
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.cifar.cifar_results import CifarResults

# Style copied from cifar_results_continual.ipynb
FINE_CLASS_COLOURS = {"12": "tab:blue", "1,2": "tab:blue", "3": "tab:orange", "4": "tab:red", "5": "tab:purple"}
REGIME_CHANGE_COLOUR = "lightgrey"
CHANCE_COLOUR = "black"
PRETRAINED_COLOUR = "brown"
PRETRAINED_VALUE = 0.311
CHANCE_VALUE = 0.5

ORDERS = [[3, 4, 5], [4, 5, 3], [5, 3, 4]]
STREAM_CLASSES = [3, 4, 5]
FINE_NAMES = ["12", "3", "4", "5"]

METHOD_LABELS = {
    "stm": "CLS/STM (RL, paper)",
    "ltm": "ResNet-18 LTM fine-tuned (paper)",
    "linear": "Linear probe head",
    "ncm": "NCM head",
    "flymodel": "FlyModel head",
    "sdmlp": "SDMLP head",
}

DEFAULT_GLOBS = {
    "continual": {
        "stm": ["runs*/cifar_100/continual_[[]*[]]_500/**/results_continual.txt",
                "runs*/cifar_100/stm*continual_[[]*[]]_500/**/results_*continual.txt"],
        "ltm": ["runs*/cifar_100/continual_[[]*[]]_500/**/results_continual.txt",
                "runs*/cifar_100/ltm*continual_[[]*[]]_500/**/results_*continual.txt"],
        **{h: [f"runs*/cifar_100/head_{h}_continual_[[]*[]]_500/**/results_continual.txt"]
           for h in ["linear", "ncm", "flymodel", "sdmlp"]},
    },
    "streaming": {
        "stm": ["runs*/cifar_100/few-shot_[[]?[]]_500/**/results_few-shot.txt",
                "runs*/cifar_100/streaming_[[]?[]]_500/**/results_streaming.txt",
                "runs*/cifar_100/stm*streaming_[[]?[]]_500/**/results_*streaming.txt"],
        "ltm": ["runs*/cifar_100/few-shot_[[]?[]]_500/**/results_few-shot.txt",
                "runs*/cifar_100/streaming_[[]?[]]_500/**/results_streaming.txt",
                "runs*/cifar_100/ltm*few-shot_[[]?[]]_500/**/results_*few-shot.txt",
                "runs*/cifar_100/ltm*streaming_[[]?[]]_500/**/results_*streaming.txt"],
        **{h: [f"runs*/cifar_100/head_{h}_streaming_[[]?[]]_500/**/results_streaming.txt"]
           for h in ["linear", "ncm", "flymodel", "sdmlp"]},
    },
}


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=str, default=".", help="Directory the glob patterns are relative to.")
    p.add_argument("--out-dir", type=str, default="runs_local/comparison")
    p.add_argument("--baselines-dir", type=str, default="runs_local/baselines",
                   help="Directory holding ltm_pretrained_baseline.txt / stm_pretrained_baseline.txt (pre-continual "
                        "accuracies R[0, j] for the reference methods; see eval_pretrained_baseline.py).")
    p.add_argument("--methods", nargs="+", default=list(METHOD_LABELS.keys()))
    p.add_argument("--glob", action="append", default=[], metavar="METHOD=PATTERN",
                   help="Override the glob(s) for a method (repeatable; may also be METHOD:streaming=PATTERN).")
    p.add_argument("--settings", nargs="+", default=["continual", "streaming"], choices=["continual", "streaming"])
    return p.parse_args()


def sniff_reference_kind(path: str) -> str | None:
    """
    Content-based STM/LTM discrimination for the paper's reference scripts, which share the experiment
    name "continual_[3, 4, 5]_500": cifar_main_ltm_fine_tuning.py writes the whole fine-class order in
    its training lines ("[3, 4, 5]"), cifar_main_stm_training.py writes the current class only ("[3]").
    Returns "ltm", "stm" or None (no training line found).
    """
    try:
        with open(path) as f:
            for line in f:
                parts = re.split(r'\s*,\s*(?=(?:[^\[\]]*\[[^\[\]]*\])*[^\[\]]*$)', line.strip())
                if len(parts) >= 3 and parts[2] == "training":
                    n = len(re.findall(r"\d+", parts[1]))
                    return "ltm" if n > 1 else "stm"
    except OSError:
        return None
    return None


def find_files(runs_root: str, patterns: list[str], method: str) -> list[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(runs_root, pat), recursive=True))
    files = sorted(f for f in set(files) if "quarantine" not in f.lower())
    if method in ("stm", "ltm"):
        # The paper's STM and LTM scripts share experiment names; see module docstring. Use the path when it
        # says which is which, otherwise sniff the file content.
        kept = []
        for f in files:
            if "head_" in f:
                continue
            low = f.lower()
            if "ltm" in low and "stm" not in low:
                kind = "ltm"
            elif "stm" in low and "ltm" not in low:
                kind = "stm"
            else:
                kind = sniff_reference_kind(f)
            if kind == method:
                kept.append(f)
        files = kept
    return files


def parse_order(path: str) -> list[int] | None:
    m = re.search(r"\[(\d+(?:,\s*\d+)*)\]", path)
    if not m:
        return None
    return [int(x) for x in m.group(1).split(",")]


def parse_seed(path: str, fallback: int) -> int:
    m = re.search(r"seed[_-]?(\d+)", path)
    return int(m.group(1)) if m else fallback


def load_pivot(path: str) -> pd.DataFrame | None:
    """Epoch x fine-class accuracy table (evaluate rows only)."""
    try:
        df = CifarResults.read_results_file(path)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Could not read {path}: {e}")
        return None
    df = df[df["Mode"] == "evaluate"]
    if df.empty:
        return None
    df = df.copy()
    df["Fine class"] = df["Fine class"].astype(str).str.replace(",", "")  # "1,2" -> "12"
    piv = df.pivot_table(index="Epoch", columns="Fine class", values="Accuracy", aggfunc="mean").sort_index()
    return piv


BASELINE_PROXY = "epoch 1 proxy"


def load_baseline(method: str, results_path: str, baselines_dir: str) -> tuple[pd.Series | None, str]:
    """
    Accuracy on each test set BEFORE the continual phase (R[0, j]). Returns (series indexed by fine name, source
    description). The series is None when no baseline file exists (the caller then falls back to the epoch-1
    evaluation, BASELINE_PROXY).
      heads: last evaluate rows of results_pretrain.txt in the run directory.
      ltm:   <baselines_dir>/ltm_pretrained_baseline.txt (eval_pretrained_baseline.py).
      stm:   <baselines_dir>/stm_pretrained_baseline.txt if present.
    """
    if method in ("ltm", "stm"):
        path = os.path.join(baselines_dir, f"{method}_pretrained_baseline.txt")
    else:
        path = os.path.join(os.path.dirname(results_path), "results_pretrain.txt")
    if not os.path.exists(path):
        return None, BASELINE_PROXY
    piv = load_pivot(path)
    if piv is None or piv.empty:
        warnings.warn(f"Baseline file {path} has no evaluate rows; using the epoch-1 proxy")
        return None, BASELINE_PROXY
    return piv.iloc[-1].reindex(FINE_NAMES), os.path.basename(path)


def collect(runs_root: str, patterns: list[str], method: str, keys: list[list[int]],
            baselines_dir: str | None = None) -> dict:
    """Returns {tuple(order_or_class): [pivot per seed]}. With baselines_dir the pre-continual baseline is attached."""
    out = {tuple(k): [] for k in keys}
    files = find_files(runs_root, patterns, method)
    for i, f in enumerate(files):
        order = parse_order(f)
        if order is None or tuple(order) not in out:
            continue
        piv = load_pivot(f)
        if piv is None:
            continue
        piv.attrs["seed"] = parse_seed(f, i)
        piv.attrs["path"] = f
        if baselines_dir is not None:
            baseline, source = load_baseline(method, f, baselines_dir)
            piv.attrs["baseline"] = baseline
            piv.attrs["baseline_source"] = source
        out[tuple(order)].append(piv)
    return out


def stack_seeds(pivots: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Align on the common epochs, return (mean, std, n_seeds)."""
    epochs = sorted(set.intersection(*[set(p.index) for p in pivots]))
    cols = [c for c in FINE_NAMES if all(c in p.columns for p in pivots)]
    arr = np.stack([p.loc[epochs, cols].to_numpy(dtype=float) for p in pivots], axis=0)
    mean = pd.DataFrame(arr.mean(0), index=epochs, columns=cols)
    std = pd.DataFrame(arr.std(0), index=epochs, columns=cols)
    return mean, std, len(pivots)


# --------------------------------------------------------------------------------------
def plot_method(method: str, setting: str, data: dict, out_dir: str):
    keys = list(data.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 6), sharey=True)
    if len(keys) == 1:
        axes = [axes]
    any_data = False
    for i, (ax, key) in enumerate(zip(axes, keys)):
        pivots = data[key]
        title = ("Trained on fine-classes " if setting == "continual" else "Trained on fine-class ") + \
                ",".join(str(k) for k in key)
        if not pivots:
            ax.set_title(title + " (no data)", fontsize=16, pad=12)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_xticks([])
            continue
        any_data = True
        mean, std, n = stack_seeds(pivots)
        for fc in mean.columns:
            ax.plot(mean.index, mean[fc], label=fc if fc != "12" else "1,2", color=FINE_CLASS_COLOURS.get(fc),
                    linewidth=2, marker="o", markersize=4)
            if n > 1:
                ax.fill_between(mean.index, mean[fc] - std[fc], mean[fc] + std[fc],
                                color=FINE_CLASS_COLOURS.get(fc), alpha=0.15, linewidth=0)
        if setting == "continual":
            n_per_phase = len(mean.index) // 3
            for x in (n_per_phase + 1, 2 * n_per_phase + 1):
                ax.axvline(x, color=REGIME_CHANGE_COLOUR, linestyle="-", linewidth=1.5)
        ax.axhline(PRETRAINED_VALUE, color=PRETRAINED_COLOUR, linestyle="--", linewidth=1.5, label="LTM (pre) 3,4,5")
        ax.axhline(CHANCE_VALUE, color=CHANCE_COLOUR, linestyle="-.", linewidth=1.5, label="Random binary")
        ax.set_title(title + (f"  (n={n} seeds)" if n > 1 else ""), fontsize=16, pad=12)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(title="Fine class", frameon=True, framealpha=1.0)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_ylim(0, 1)
    fig.suptitle(f"{METHOD_LABELS.get(method, method)} - {setting}", fontsize=18)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{method}_{setting}.png")
    if any_data:
        fig.savefig(path, dpi=120)
        print(f"wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Continual-learning metrics (Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018), adapted to this protocol
# --------------------------------------------------------------------------------------
CL_METRICS = ["ACC", "ACC_new", "BWT", "BWT_0", "FWT", "forgetting_eop", "LA"]
CL_METRIC_LABELS = {
    "forgetting_eop": "Forgetting (end-of-phase)",
    "forgetting_mean": "Forgetting (per-epoch max)",
}
CL_METRIC_DEFINITIONS = (
    "Continual-learning metrics (Lopez-Paz & Ranzato 2017 / Chaudhry et al. 2018, adapted). Phases t = 1..T (T = 3) "
    "in the run's fine-class order; test sets j = 0..T where j = 0 is fine-classes 1,2 (pre-trained on) and j = t is "
    "the fine class trained in phase t. R[t, j] = accuracy on test set j after the LAST epoch of phase t (epochs 12, "
    "24, 36); R[0, j] = accuracy BEFORE the continual phase (heads: last evaluate rows of results_pretrain.txt; LTM: "
    "eval_pretrained_baseline.py output; STM: stm_pretrained_baseline.txt or, failing that, its epoch-1 evaluation as a "
    "proxy, footnoted). "
    "ACC = mean_{j=0..T} R[T, j] (= final_mean). "
    "ACC_new = mean_{j=1..T} R[T, j] (continually learned classes only). "
    "BWT = mean_{j=1..T-1} (R[T, j] - R[j, j]) (negative = forgetting). "
    "BWT_0 = R[T, 0] - R[0, 0] (drift on the pre-trained classes). "
    "FWT = mean_{j=2..T} (R[j-1, j] - R[0, j]) (the label space is fixed in this domain-incremental setting, so "
    "R[j-1, j] is well defined). "
    "Forgetting (end-of-phase) = mean_{j=1..T-1} (max_{l in j..T-1} R[l, j] - R[T, j]). "
    "Forgetting (per-epoch max) = mean over the trained classes of (max accuracy on the class at ANY epoch from the "
    "start of its own phase onward) minus its final accuracy (the pre-existing forgetting_mean column). "
    "LA = mean_{j=1..T} R[j, j] (learning accuracy / plasticity). "
    "Rows with order 'mean' pool every seed of every order of the method (mean ± std over those runs). "
    "The full R matrices are in R_matrices.csv."
)


def r_matrix(piv: pd.DataFrame, order: tuple[int, ...]) -> pd.DataFrame:
    """
    Accuracy matrix R[t, j]: rows = phase 0..T (0 = before the continual phase), columns = FINE_NAMES.
    Row t >= 1 is the evaluation after the last epoch of phase t. Row 0 is piv.attrs["baseline"] when present,
    otherwise the epoch-1 evaluation (R.attrs["baseline_source"] == BASELINE_PROXY).
    """
    T = len(order)
    n_per_phase = len(piv.index) // T
    baseline = piv.attrs.get("baseline")
    source = piv.attrs.get("baseline_source", BASELINE_PROXY)
    if baseline is None:
        baseline = piv.iloc[0].reindex(FINE_NAMES)
        source = BASELINE_PROXY
    rows = [baseline.reindex(FINE_NAMES).to_numpy(dtype=float)]
    for t in range(1, T + 1):
        rows.append(piv.iloc[t * n_per_phase - 1].reindex(FINE_NAMES).to_numpy(dtype=float))
    R = pd.DataFrame(rows, index=pd.RangeIndex(T + 1, name="phase"), columns=FINE_NAMES)
    R.attrs["baseline_source"] = source
    R.attrs["trained_class"] = ["12"] + [str(c) for c in order]
    return R


def cl_metrics(R: pd.DataFrame, order: tuple[int, ...]) -> dict:
    """
    ACC, ACC_new, BWT, BWT_0, FWT, forgetting_eop, LA from the R matrix (see CL_METRIC_DEFINITIONS).
    Test set j maps to column "12" for j = 0 and str(order[j - 1]) for j >= 1. In this domain-incremental
    setting all phases share one 20-way label space, so R[j-1, j] (accuracy on a class before it is trained
    on) is well defined and FWT is measured against the pre-continual baseline R[0, j].
    """
    T = len(order)
    col = ["12"] + [str(c) for c in order]

    def r(t, j):
        return float(R.loc[t, col[j]])

    m = {}
    m["ACC"] = float(np.nanmean([r(T, j) for j in range(T + 1)]))
    m["ACC_new"] = float(np.nanmean([r(T, j) for j in range(1, T + 1)]))
    m["BWT"] = float(np.nanmean([r(T, j) - r(j, j) for j in range(1, T)])) if T > 1 else np.nan
    m["BWT_0"] = r(T, 0) - r(0, 0)
    m["FWT"] = float(np.nanmean([r(j - 1, j) - r(0, j) for j in range(2, T + 1)])) if T > 1 else np.nan
    m["forgetting_eop"] = float(np.nanmean([max(r(l, j) for l in range(j, T)) - r(T, j)
                                            for j in range(1, T)])) if T > 1 else np.nan
    m["LA"] = float(np.nanmean([r(j, j) for j in range(1, T + 1)]))
    return m


def aggregate_rows(per_seed: list[dict], method: str, order_label: str, cols: list[str]) -> dict:
    df = pd.DataFrame(per_seed)
    row = {"method": method, "order": order_label, "n_seeds": len(df)}
    for col in cols:
        if col in df:
            row[col] = df[col].mean()
            row[col + "_std"] = df[col].std(ddof=0) if len(df) > 1 else 0.0
    sources = sorted(set(str(r.get("baseline_source")) for r in per_seed))
    row["baseline"] = ";".join(sources)
    return row


def summarise_continual(method: str, data: dict, r_rows: list[dict] | None = None) -> list[dict]:
    """
    One summary row per order plus one row (order "mean") pooling every seed of every order. If r_rows is
    given, the R matrix of every run is appended to it (one record per phase) for R_matrices.csv.
    """
    rows = []
    all_seeds = []
    agg_cols = [f"final_{fc}" for fc in FINE_NAMES] + ["final_mean", "forgetting_3", "forgetting_4", "forgetting_5",
                                                       "forgetting_mean", "forgetting_12"] + CL_METRICS
    for order, pivots in data.items():
        if not pivots:
            continue
        per_seed = []
        for piv in pivots:
            epochs = list(piv.index)
            n_per_phase = len(epochs) // 3
            final = piv.iloc[-1]
            rec = {"seed": piv.attrs.get("seed")}
            for fc in FINE_NAMES:
                rec[f"final_{fc}"] = float(final[fc]) if fc in piv.columns else np.nan
            rec["final_mean"] = float(np.nanmean([rec[f"final_{fc}"] for fc in FINE_NAMES]))
            # forgetting (per-epoch max): max accuracy on class c at any epoch from the start of its own phase
            # onward, minus final accuracy
            forgets = []
            for i, fc in enumerate(order):
                col = str(fc)
                if col not in piv.columns or n_per_phase == 0:
                    continue
                own_and_after = piv[col].iloc[i * n_per_phase:]
                forgets.append(float(own_and_after.max() - final[col]))
                rec[f"forgetting_{col}"] = forgets[-1]
            rec["forgetting_mean"] = float(np.mean(forgets)) if forgets else np.nan
            rec["forgetting_12"] = float(piv["12"].max() - final["12"]) if "12" in piv.columns else np.nan
            # standard CL metrics from the end-of-phase accuracy matrix
            if n_per_phase > 0:
                R = r_matrix(piv, order)
                rec.update(cl_metrics(R, order))
                rec["baseline_source"] = R.attrs["baseline_source"]
                if r_rows is not None:
                    for t in R.index:
                        r_rows.append({"method": method, "order": ",".join(map(str, order)), "seed": rec["seed"],
                                       "phase": int(t), "trained_class": R.attrs["trained_class"][t],
                                       **{fc: R.loc[t, fc] for fc in FINE_NAMES},
                                       "baseline_source": R.attrs["baseline_source"],
                                       "path": piv.attrs.get("path")})
            per_seed.append(rec)
        rows.append(aggregate_rows(per_seed, method, ",".join(map(str, order)), agg_cols))
        all_seeds.extend(per_seed)
    if len(rows) > 1:
        rows.append(aggregate_rows(all_seeds, method, "mean", agg_cols))
    return rows


def summarise_streaming(method: str, data: dict) -> list[dict]:
    rows = []
    for (fc,), pivots in data.items():
        if not pivots:
            continue
        per_seed = []
        for piv in pivots:
            final = piv.iloc[-1]
            rec = {f"final_{n}": float(final[n]) if n in piv.columns else np.nan for n in FINE_NAMES}
            rec["final_mean"] = float(np.nanmean(list(rec.values())))
            rec[f"max_{fc}"] = float(piv[str(fc)].max()) if str(fc) in piv.columns else np.nan
            rec["forgetting_12"] = float(piv["12"].max() - final["12"]) if "12" in piv.columns else np.nan
            per_seed.append(rec)
        df = pd.DataFrame(per_seed)
        row = {"method": method, "trained_class": fc, "n_seeds": len(df)}
        for col in df.columns:
            row[col] = df[col].mean()
            row[col + "_std"] = df[col].std(ddof=0) if len(df) > 1 else 0.0
        rows.append(row)
    return rows


PROXY_FOOTNOTE_COLS = ("FWT", "BWT_0")


def to_markdown(df: pd.DataFrame, cols: list[str], labels: dict | None = None) -> str:
    """Markdown table; FWT / BWT_0 cells are marked with a dagger when the run used the epoch-1 baseline proxy."""
    cols = [c for c in cols if c in df.columns]
    labels = labels or {}
    lines = ["| " + " | ".join(labels.get(c, c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        proxy = BASELINE_PROXY in str(r.get("baseline", ""))
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                s = f"{v:.3f}"
                if c + "_std" in df.columns and r.get("n_seeds", 1) > 1:
                    s += f" ± {r[c + '_std']:.3f}"
                if proxy and c in PROXY_FOOTNOTE_COLS:
                    s += " †"
                cells.append(s)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    overrides = {"continual": {}, "streaming": {}}
    for item in args.glob:
        key, pattern = item.split("=", 1)
        if ":" in key:
            method, setting = key.split(":", 1)
            overrides[setting].setdefault(method, []).append(pattern)
        else:
            for setting in overrides:
                overrides[setting].setdefault(key, []).append(pattern)

    for setting in args.settings:
        keys = ORDERS if setting == "continual" else [[c] for c in STREAM_CLASSES]
        summary_rows = []
        r_rows = []
        for method in args.methods:
            patterns = overrides[setting].get(method) or DEFAULT_GLOBS[setting].get(method)
            if not patterns:
                warnings.warn(f"No glob patterns for method {method!r}; skipping")
                continue
            data = collect(args.runs_root, patterns, method, keys,
                           baselines_dir=args.baselines_dir if setting == "continual" else None)
            n_files = sum(len(v) for v in data.values())
            if n_files == 0:
                warnings.warn(f"[{setting}] no results found for method {method!r} (patterns: {patterns}); skipping")
                continue
            print(f"[{setting}] {method}: " + ", ".join(f"{','.join(map(str, k))}: {len(v)} seed(s)" for k, v in data.items()))
            plot_method(method, setting, data, args.out_dir)
            summary_rows.extend(summarise_continual(method, data, r_rows) if setting == "continual"
                                else summarise_streaming(method, data))

        if not summary_rows:
            print(f"[{setting}] nothing to summarise")
            continue
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(args.out_dir, f"summary_{setting}.csv")
        df.to_csv(csv_path, index=False)
        if setting == "continual":
            cols = ["method", "order", "n_seeds", "final_12", "final_3", "final_4", "final_5", "final_mean",
                    "ACC", "ACC_new", "BWT", "BWT_0", "FWT", "forgetting_eop", "forgetting_mean", "LA",
                    "forgetting_12"]
            if r_rows:
                r_path = os.path.join(args.out_dir, "R_matrices.csv")
                pd.DataFrame(r_rows).to_csv(r_path, index=False)
                print(f"wrote {r_path}")
        else:
            cols = ["method", "trained_class", "n_seeds", "final_12", "final_3", "final_4", "final_5", "final_mean",
                    "forgetting_12"]
        md = to_markdown(df, cols, CL_METRIC_LABELS)
        md_path = os.path.join(args.out_dir, f"summary_{setting}.md")
        with open(md_path, "w") as f:
            f.write(f"# {setting} summary\n\n")
            f.write("final_*: accuracy after the last epoch (mean over seeds, ± std when >1 seed). "
                    "forgetting_mean: average over the trained classes of (max accuracy on the class from the start "
                    "of its own training phase onward) minus its final accuracy. forgetting_12: max accuracy on "
                    "fine-classes 1,2 over the run minus final.\n\n")
            if setting == "continual":
                f.write(CL_METRIC_DEFINITIONS + "\n\n")
            f.write(md + "\n")
            if setting == "continual" and BASELINE_PROXY in " ".join(df.get("baseline", pd.Series(dtype=str)).astype(str)):
                f.write(f"\n† baseline = {BASELINE_PROXY}: no pre-continual evaluation was available, so R[0, j] is the "
                        "evaluation after the first continual epoch (already one epoch into phase 1); FWT and BWT_0 are therefore "
                        "measured relative to that proxy rather than to the true pre-continual accuracy.\n")
        print(f"wrote {csv_path} and {md_path}")
        print(md)


if __name__ == "__main__":
    main()
