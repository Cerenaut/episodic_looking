"""
Tables of results_v2.tex from runs_v2/ (Notes/experiments/plan.md, code item C4).

Reads the tree written by run_v2.sh:

    runs_v2/<setting>/<model>/pair<a>_<b>/seed<k>/<unit>/.../results_*.txt
    runs_v2/baseline/<stm model>/pair<a>_<b>/seed<k>/.../results_evaluate[_val].txt   (STM starting points)
    runs_v2/baseline/ltm/pair<a>_<b>/results_evaluate.txt                            (LTM starting points)

and prints, as LaTeX tabular rows (or markdown with --format md), the tables of results_v2.tex:

    continual   per pair: Model (n) & ACC & LA & BWT & BWT_0 & FWT & Forgetting & epochs/phase   (tab:cl-main)
    starting    per pair: accuracy on the four test sets before the continual phases              (tab:start)
    pairs       ACC by pair A-D                                                                     (tab:cl-pairs)
    stream      trained class fine 3/4/5, fine 1,2 and the average of the four sets, final        (tab:streaming)
    fewshot     trained-class accuracy and the four-set average per N, averaged over fine 3-5     (tab:fewshot)
    e40         continual ACC on the e40 LTM                                                         (tab:ablations)

Metric definitions are plot_comparison.py's (r_matrix, cl_metrics), so the numbers match the existing tables.
Continual metrics are computed per run (seed x order); a model's value is the mean over seeds of each seed's mean
over its orders, and the +/- is the standard deviation over seeds. Forgetting is forgetting_eop. Missing cells print
\\pending, so the output can be pasted over the table bodies.

--part val reads the validation files (results_*_val.txt) instead, for choosing budgets; --plateau then prints, per
model and setting, the validation accuracy of the trained class against epochs (fine 3-5 only, averaged over seeds,
classes, orders and phases, by epoch within the phase) and the budget the protocol picks: the smallest epoch within
0.01 of the maximum.

Usage: python metrics_v2.py [--runs runs_v2] [--tables continual stream ...] [--part test|val] [--plateau]
       [--plateau-pair pair0_1] [--partial] [--format latex|md]
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from plot_comparison import FINE_NAMES, cl_metrics, load_pivot, r_matrix

PARTIAL = False  # --partial: settle for the fine classes that have run (cells marked *)


def load_run_pivot(path: str) -> pd.DataFrame | None:
    """
    Epoch x test-set accuracy of one run. Epochs are numbered by counting training lines, 1, 2, ..., so they are
    right whether a script numbers epochs globally (STM, heads) or restarts every phase (LTM-only, whose restarts
    CifarResults.read_results_file cannot see at 1 epoch per phase), and when evaluation is only every k-th epoch.
    """
    rows, epoch = [], 0
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in re.split(r",\s*(?![^\[]*\])", line.strip())]
            if len(parts) < 5:
                continue
            if parts[2] == "training":
                epoch += 1
            elif parts[2] == "evaluate":
                name = re.sub(r"[\[\]', ]", "", parts[1])
                rows.append((epoch, name, float(parts[4])))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["Epoch", "Fine class", "Accuracy"])
    return df.pivot_table(index="Epoch", columns="Fine class", values="Accuracy", aggfunc="mean").sort_index()

MODELS = [
    ("rl", "CLS/STM, RL"),
    ("actor", "CLS/STM, diff.\\ actor"),
    ("linear", "Linear probe"),
    ("ncm", "NCM"),
    ("flymodel", "FlyModel"),
    ("sdmlp", "SDMLP"),
    ("ltm", "LTM-only fine-tuning"),
]
PAIRS = [("A", "pair0_1"), ("B", "pair2_3"), ("C", "pair5_6"), ("D", "pair15_16")]
ORDERS = [(3, 4, 5), (4, 5, 3), (5, 3, 4)]
FEWSHOT_N = [1, 4, 16, 64, 400]
PENDING = "\\pending"


# --------------------------------------------------------------------------------------
def results_files(unit_dir: str, part: str) -> list[str]:
    """The run's own results file(s): not pre-training, not per-image records; _val for part val."""
    out = []
    for f in glob.glob(os.path.join(glob.escape(unit_dir), "**", "results_*.txt"), recursive=True):
        name = os.path.basename(f)
        if name.startswith(("results_pretrain", "results_evaluate")) or name.endswith("_images.txt"):
            continue
        if name.endswith("_val.txt") == (part == "val"):
            out.append(f)
    return sorted(out)


def one_file(unit_dir: str, part: str) -> str | None:
    files = results_files(unit_dir, part)
    if len(files) > 1:
        warnings.warn(f"{len(files)} results files under {unit_dir}; using the newest")
    return files[-1] if files else None


def seeds_of(path: str) -> list[tuple[int, str]]:
    out = []
    for d in sorted(glob.glob(os.path.join(glob.escape(path), "seed*"))):
        m = re.fullmatch(r"seed(\d+)", os.path.basename(d))
        if m and os.path.isdir(d):
            out.append((int(m.group(1)), d))
    return out


def finished(unit_dir: str) -> bool:
    """run_v2.sh marks a finished unit with job.done; unfinished units are left out, and reported."""
    if os.path.exists(os.path.join(unit_dir, "job.done")):
        return True
    if os.path.isdir(unit_dir):
        print(f"% skipping unfinished {unit_dir}", file=sys.stderr)
    return False


def baseline_series(runs: str, setting_suffix: str, model: str, pair: str, seed: int, unit_dir: str,
                    part: str) -> pd.Series | None:
    """Accuracy on the four sets before the continual phases (R[0, j])."""
    suffix = "_val" if part == "val" else ""
    if model in ("rl", "actor"):
        d = os.path.join(runs, f"baseline{setting_suffix}", model, pair, f"seed{seed}")
        files = glob.glob(os.path.join(glob.escape(d), "**", f"results_evaluate{suffix}.txt"), recursive=True)
    elif model == "ltm":
        d = os.path.join(runs, f"baseline{setting_suffix}", "ltm", pair)
        files = glob.glob(os.path.join(glob.escape(d), f"results_evaluate{suffix}.txt"))
    else:
        files = glob.glob(os.path.join(glob.escape(unit_dir), "**", f"results_pretrain{suffix}.txt"), recursive=True)
    if not files:
        return None
    piv = load_pivot(sorted(files)[-1])
    return None if piv is None or piv.empty else piv.iloc[-1].reindex(FINE_NAMES)


# --------------------------------------------------------------------------------------
def continual_runs(runs: str, setting: str, model: str, pair: str, part: str) -> dict[int, list[dict]]:
    """{seed: [metrics per order]}"""
    out = {}
    suffix = setting[len("continual"):]
    for seed, seed_dir in seeds_of(os.path.join(runs, setting, model, pair)):
        for order in ORDERS:
            unit = os.path.join(seed_dir, "order" + "_".join(map(str, order)))
            if not finished(unit):
                continue
            f = one_file(unit, part)
            piv = load_run_pivot(f) if f else None
            if piv is None or len(piv.index) % 3:
                warnings.warn(f"no complete results in {unit}")
                continue
            base = baseline_series(runs, suffix, model, pair, seed, unit, part)
            piv.attrs["baseline"] = base
            piv.attrs["baseline_source"] = "file" if base is not None else "epoch 1 proxy"
            R = r_matrix(piv, order)
            m = cl_metrics(R, order)
            m["epochs_per_phase"] = len(piv.index) // 3
            m["baseline_proxy"] = base is None
            out.setdefault(seed, []).append(m)
    return out


def seed_stats(per_seed: dict[int, list[dict]], key: str) -> tuple[float, float, int]:
    vals = [np.mean([m[key] for m in ms]) for ms in per_seed.values() if ms]
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def fmt(x: float, signed: bool = False, sd: float | None = None, fmt_kind: str = "latex") -> str:
    if x is None or not np.isfinite(x):
        return PENDING if fmt_kind == "latex" else "to run"
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    if fmt_kind == "latex" and (signed or s.startswith("-")):
        s = f"${s}$"  # as the tables write signed values: $+0.034$, $-0.005$
    if sd is not None and np.isfinite(sd):
        s += f"\\sd{{{sd:.3f}}}" if fmt_kind == "latex" else f" ± {sd:.3f}"
    return s


def emit(cells: list[str], fmt_kind: str) -> str:
    return ("    " + " & ".join(cells) + " \\\\") if fmt_kind == "latex" else ("| " + " | ".join(cells) + " |")


def table_continual(runs, part, fmt_kind, setting="continual", pairs=PAIRS, models=MODELS):
    lines = []
    for letter, pair in pairs:
        lines.append(f"% {setting}, pair {letter} ({pair}), part {part}: Model (n) & ACC & LA & BWT & BWT_0 & FWT & "
                     f"Forgetting & epochs/phase")
        for model, label in models:
            per_seed = continual_runs(runs, setting, model, pair, part)
            acc, acc_sd, n = seed_stats(per_seed, "ACC")
            cells = [f"{label} ({n})", fmt(acc, sd=acc_sd if n > 1 else None, fmt_kind=fmt_kind)]
            for key, signed in (("LA", False), ("BWT", True), ("BWT_0", True), ("FWT", True), ("forgetting_eop", False)):
                cells.append(fmt(seed_stats(per_seed, key)[0], signed=signed, fmt_kind=fmt_kind))
            epochs = sorted({m["epochs_per_phase"] for ms in per_seed.values() for m in ms})
            cells.append(",".join(map(str, epochs)) if epochs else "-")
            if any(m["baseline_proxy"] for ms in per_seed.values() for m in ms):
                cells[0] += " (no baseline file: BWT_0, FWT from the epoch-1 proxy)"
            lines.append(emit(cells, fmt_kind))
    return lines


def table_starting(runs, part, fmt_kind):
    lines = []
    for letter, pair in PAIRS:
        lines.append(f"% starting points, pair {letter}, part {part}: Model & Fine 1, 2 & Fine 3 & Fine 4 & Fine 5")
        for model, label in MODELS:
            series = []
            if model in ("rl", "actor"):
                for seed, _ in seeds_of(os.path.join(runs, "baseline", model, pair)):
                    s = baseline_series(runs, "", model, pair, seed, "", part)
                    if s is not None:
                        series.append(s)
            elif model == "ltm":
                s = baseline_series(runs, "", "ltm", pair, 0, "", part)
                if s is not None:
                    series.append(s)
            else:
                for seed, seed_dir in seeds_of(os.path.join(runs, "continual", model, pair)):
                    unit = os.path.join(seed_dir, "order3_4_5")
                    if finished(unit):
                        s = baseline_series(runs, "", model, pair, seed, unit, part)
                        if s is not None:
                            series.append(s)
            cells = [f"{label} ({len(series)})"]
            for name in FINE_NAMES:
                cells.append(fmt(float(np.mean([s[name] for s in series])) if series else np.nan, fmt_kind=fmt_kind))
            lines.append(emit(cells, fmt_kind))
    return lines


def table_pairs(runs, part, fmt_kind):
    lines = [f"% continual ACC by pair, part {part}: Model & A & B & C & D"]
    for model, label in MODELS:
        cells = [label]
        for _, pair in PAIRS:
            acc, sd, n = seed_stats(continual_runs(runs, "continual", model, pair, part), "ACC")
            cells.append(fmt(acc, sd=sd if n > 1 else None, fmt_kind=fmt_kind) + (f" ({n})" if n else ""))
        lines.append(emit(cells, fmt_kind))
    return lines


def final_rows(runs, setting, model, pair, units: list[str], part) -> dict[str, list[pd.Series]]:
    """{unit: [final evaluation row per seed]}"""
    out = {u: [] for u in units}
    for seed, seed_dir in seeds_of(os.path.join(runs, setting, model, pair)):
        for u in units:
            unit = os.path.join(seed_dir, u)
            if not finished(unit):
                continue
            f = one_file(unit, part)
            piv = load_run_pivot(f) if f else None
            if piv is not None and not piv.empty:
                piv = piv.reindex(columns=FINE_NAMES)
                row = piv.iloc[-1].copy()
                row["epochs"] = int(piv.index[-1])
                out[u].append(row)
    return out


def enough(values: list, needed: int = 3) -> bool:
    return len(values) == needed or (PARTIAL and len(values) > 0)


def mark(cell: str, values: list, needed: int = 3) -> str:
    return cell + "*" if PARTIAL and 0 < len(values) < needed else cell


def table_stream(runs, part, fmt_kind, pair="pair0_1"):
    lines = [f"% single-stream, {pair}, part {part}: Model & Fine 3 & Fine 4 & Fine 5 (trained class) & Fine 1, 2 & "
             f"Average & epochs"]
    for model, label in MODELS:
        rows = final_rows(runs, "stream", model, pair, ["fine3", "fine4", "fine5"], part)
        cells = [label]
        for c in (3, 4, 5):
            vals = [r[str(c)] for r in rows[f"fine{c}"]]
            cells.append(fmt(np.mean(vals) if vals else np.nan, sd=np.std(vals) if len(vals) > 1 else None,
                             fmt_kind=fmt_kind))
        allrows = [r for c in (3, 4, 5) for r in rows[f"fine{c}"]]
        # fine 1,2 and the average: mean over trained classes of the mean over seeds
        per_class_12 = [np.mean([r["12"] for r in rows[f"fine{c}"]]) for c in (3, 4, 5) if rows[f"fine{c}"]]
        per_class_avg = [np.mean([r[FINE_NAMES].mean() for r in rows[f"fine{c}"]]) for c in (3, 4, 5) if rows[f"fine{c}"]]
        cells.append(mark(fmt(np.mean(per_class_12) if enough(per_class_12) else np.nan, fmt_kind=fmt_kind), per_class_12))
        cells.append(mark(fmt(np.mean(per_class_avg) if enough(per_class_avg) else np.nan, fmt_kind=fmt_kind), per_class_avg))
        cells.append(",".join(map(str, sorted({int(r["epochs"]) for r in allrows}))) or "-")
        lines.append(emit(cells, fmt_kind))
    return lines


def table_fewshot(runs, part, fmt_kind, pair="pair0_1", ns=FEWSHOT_N):
    lines = [f"% few-shot, {pair}, part {part}, mean over fine classes 3-5 of the mean over seeds: Model & "
             + " & ".join(f"trained N={n}" for n in ns) + " & " + " & ".join(f"average N={n}" for n in ns)]
    for model, label in MODELS:
        trained, average = [], []
        for n in ns:
            rows = final_rows(runs, "fewshot", model, pair, [f"fine{c}_n{n}" for c in (3, 4, 5)], part)
            t = [np.mean([r[str(c)] for r in rows[f"fine{c}_n{n}"]]) for c in (3, 4, 5) if rows[f"fine{c}_n{n}"]]
            a = [np.mean([r[FINE_NAMES].mean() for r in rows[f"fine{c}_n{n}"]]) for c in (3, 4, 5) if rows[f"fine{c}_n{n}"]]
            trained.append(mark(fmt(np.mean(t) if enough(t) else np.nan, fmt_kind=fmt_kind), t))
            average.append(mark(fmt(np.mean(a) if enough(a) else np.nan, fmt_kind=fmt_kind), a))
        lines.append(emit([label] + trained + average, fmt_kind))
    return lines


def table_e40(runs, part, fmt_kind):
    return table_continual(runs, part, fmt_kind, setting="continual_e40", pairs=[PAIRS[0]])


# --------------------------------------------------------------------------------------
def plateau(runs, fmt_kind, pair="pair0_1", ns=FEWSHOT_N):
    """Validation curves of the trained class and the protocol's budget: smallest epoch within 0.01 of the max."""
    lines = [f"% plateau rule on validation (fine 3-5 only), {pair}: model & setting & budget (epochs) & max & curve"]
    settings = [("continual", ["order" + "_".join(map(str, o)) for o in ORDERS]), ("stream", ["fine3", "fine4", "fine5"])]
    settings += [(f"fewshot N={n}", [f"fine{c}_n{n}" for c in (3, 4, 5)]) for n in ns]
    for model, label in MODELS:
        for setting_label, units in settings:
            setting = setting_label.split()[0]
            curves = []
            for seed, seed_dir in seeds_of(os.path.join(runs, setting, model, pair)):
                for u in units:
                    unit = os.path.join(seed_dir, u)
                    if not finished(unit):
                        continue
                    f = one_file(unit, "val")
                    piv = load_run_pivot(f) if f else None
                    if piv is None:
                        continue
                    if setting == "continual":
                        order = [int(x) for x in u[len("order"):].split("_")]
                        per = len(piv.index) // 3
                        for t, c in enumerate(order):
                            seg = piv[str(c)].iloc[t * per:(t + 1) * per].to_numpy()
                            curves.append(pd.Series(seg, index=range(1, per + 1)))
                    else:
                        c = u[len("fine"):].split("_")[0]
                        curves.append(piv[c])
            if not curves:
                continue
            curve = pd.concat(curves, axis=1).mean(axis=1).dropna()
            budget = int(curve.index[np.argmax(curve.to_numpy() >= curve.max() - 0.01)])
            points = " ".join(f"{e}:{v:.3f}" for e, v in curve.items())
            lines.append(emit([label, setting_label, str(budget), f"{curve.max():.3f}", points], fmt_kind))
    return lines


TABLES = {"continual": table_continual, "starting": table_starting, "pairs": table_pairs, "stream": table_stream,
          "fewshot": table_fewshot, "e40": table_e40}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs_v2")
    p.add_argument("--tables", nargs="+", default=list(TABLES), choices=list(TABLES))
    p.add_argument("--part", choices=["test", "val"], default="test")
    p.add_argument("--format", choices=["latex", "md"], default="latex")
    p.add_argument("--plateau", action="store_true")
    p.add_argument("--plateau-pair", default="pair0_1")
    p.add_argument("--partial", action="store_true",
                   help="Single-stream and few-shot: average over the fine classes that have run (cells marked *), "
                        "e.g. for the pilot on fine class 3 only.")
    args = p.parse_args()
    global PARTIAL
    PARTIAL = args.partial
    if not os.path.isdir(args.runs):
        sys.exit(f"no such directory: {args.runs}")
    warnings.simplefilter("always")
    for name in args.tables:
        for line in TABLES[name](args.runs, args.part, args.format):
            print(line)
        print()
    if args.plateau:
        for line in plateau(args.runs, args.format, args.plateau_pair):
            print(line)


if __name__ == "__main__":
    main()
