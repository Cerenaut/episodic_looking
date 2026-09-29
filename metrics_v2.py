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
model and setting, the selection curve against epochs (validation_curve: continual, the new classes learnt so far;
single-stream and few-shot, the mean of the trained class and fine 1,2, which counts retention) and the budget the
protocol's rule picks (selection_rule).

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
    """Mean and sd over seeds of each seed's mean over its orders. Only seeds with all three orders count, so that a
    half-finished seed does not weigh as much as a complete one (with --partial, any finished order counts)."""
    vals = [np.mean([m[key] for m in ms]) for ms in per_seed.values() if len(ms) == len(ORDERS) or (PARTIAL and ms)]
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
SMOOTH_POINTS = 3      # centred running mean over this many evaluations (truncated at the ends) before the rule
MIN_DELTA = 0.01       # an improvement must exceed the best so far by more than this; also the budget tolerance
PATIENCE = 0.5         # plateaued once the run has gone this fraction longer than the epoch of its last improvement
TIE = 0.03             # learning rates within this of the best are tied (about one standard error of a 200-image set)
SELECTION_CLASSES = (3, 4, 5)   # single-stream and few-shot selection average fine classes 3, 4 and 5
# Single-stream and few-shot: the selection curve is the mean of the trained class and fine 1,2 (retention), not the
# trained class alone (decided 29 Sep: a learning-only curve chose budgets that cost fine 1,2 more than the new class
# gained). Fine 1,2 validation is optimistic (the LTM was pre-trained on those images), so forgetting is understated.
RETENTION_WEIGHT = 0.5
GRID_EXTENSION_MARK = "grid_extension"  # file in a tree created by a grid-edge step (only one step is taken)


def selection_rule(curve: pd.Series) -> dict:
    """
    The protocol's budget rule on one validation curve (index = epoch, 1-based):
      smooth (centred running mean of SMOOTH_POINTS evaluations, truncated at the ends);
      last improvement = last epoch at which the smoothed value beat the best so far by more than MIN_DELTA;
      plateaued if the run is at least (1 + PATIENCE) x that epoch long; otherwise extend (the run is doubled);
      budget = smallest epoch whose smoothed value is within MIN_DELTA of the best smoothed value;
      unresolved if that is the first evaluation and the next one is more than one epoch later (the peak may lie
      between them).
    """
    smooth = curve.rolling(SMOOTH_POINTS, center=True, min_periods=1).mean()
    best, last_improvement = -np.inf, int(smooth.index[0])
    for epoch, value in smooth.items():
        if value > best + MIN_DELTA:
            best, last_improvement = value, int(epoch)
    top = smooth.max()
    budget = int(smooth.index[np.argmax(smooth.to_numpy() >= top - MIN_DELTA)])
    length = int(smooth.index[-1])
    unresolved = budget == int(smooth.index[0]) and len(smooth) > 1 and int(smooth.index[1]) - budget > 1
    plateaued = length >= (1 + PATIENCE) * last_improvement
    return {"budget": budget, "value": float(smooth.loc[budget]), "best": float(top),
            "last_improvement": last_improvement, "length": length, "plateaued": plateaued, "unresolved": unresolved}


def selection_units(setting: str, n: int | None) -> list[str]:
    if setting == "continual":
        return ["order" + "_".join(map(str, o)) for o in ORDERS]
    if setting == "stream":
        return [f"fine{c}" for c in SELECTION_CLASSES]
    return [f"fine{c}_n{n}" for c in SELECTION_CLASSES]


def validation_curve(runs, model, setting, n, pair) -> pd.Series | None:
    """
    Selection curve on the validation split, from every unit of the setting (None unless all have finished, or with
    --partial any): continual, the accuracy over the fine classes 3-5 seen so far, by epoch within the phase, averaged
    over phases and orders; single-stream and few-shot, (1 - RETENTION_WEIGHT) x the accuracy on the trained class +
    RETENTION_WEIGHT x the accuracy on fine 1,2, averaged over fine classes 3, 4, 5. Only epochs evaluated in every unit are kept, so units with different evaluation schedules are not mixed.
    """
    curves, missing = [], 0
    for seed, seed_dir in seeds_of(os.path.join(runs, setting, model, pair)) or [(1, None)]:
        for u in selection_units(setting, n):
            unit = None if seed_dir is None else os.path.join(seed_dir, u)
            f = one_file(unit, "val") if unit and finished(unit) else None
            piv = load_run_pivot(f) if f else None
            if piv is None:
                missing += 1
                continue
            if setting == "continual":
                order = [int(x) for x in u[len("order"):].split("_")]
                per = int(piv.index.max()) // 3  # epochs per phase (the last epoch of every phase is evaluated)
                for t in range(3):
                    rows = piv[(piv.index > t * per) & (piv.index <= (t + 1) * per)]
                    seen = [str(c) for c in order[:t + 1]]
                    curves.append(pd.Series(rows[seen].mean(axis=1).to_numpy(), index=rows.index - t * per))
            else:
                trained = piv[u[len("fine"):].split("_")[0]]
                curves.append(((1 - RETENTION_WEIGHT) * trained + RETENTION_WEIGHT * piv["12"]).dropna())
    if not curves or (missing and not PARTIAL):
        return None
    return pd.concat(curves, axis=1).dropna().mean(axis=1)


def draft_budget(model: str, setting: str, n: int | None) -> int:
    """The draft's budget in the model's own epochs (results_v2.tex, table tab:budgets)."""
    stm = model in ("rl", "actor")
    if setting == "continual":
        return 12
    if setting == "stream":
        return 192 if stm else 12
    return 12 if stm else 6250 // n


def cap(model: str, setting: str, n: int | None) -> int:
    """Longest a selection run may be extended to: 4x the draft's budget (CLS/STM), 32x (heads, LTM-only)."""
    return (4 if model in ("rl", "actor") else 32) * draft_budget(model, setting, n)


def tree_lr(tree: str) -> float | None:
    m = re.search(r"_lr([0-9.eE+-]+)$", tree)
    return float(m.group(1)) if m else None


def fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def choose_learning_rate(candidates: list[dict]) -> dict:
    """
    candidates: dicts with tree, lr (None for models without one), value, extension (a grid-edge tree).
    Returns {"tree": chosen} or {"extend": new_lr, "from": tree}. Learning rates within TIE of the best are tied, and
    among tied ones the interior of the grid is preferred. The grid is extended one step only when the best is at an
    edge and beats its inner neighbour by more than TIE, and the edge is not itself an extension.
    """
    best = max(candidates, key=lambda c: c["value"])
    with_lr = sorted((c for c in candidates if c["lr"] is not None), key=lambda c: c["lr"])
    if len(with_lr) < 2 or best["lr"] is None:
        return {"tree": best["tree"]}
    lrs = [c["lr"] for c in with_lr]
    k = lrs.index(best["lr"])
    if k in (0, len(lrs) - 1):
        inner = with_lr[1] if k == 0 else with_lr[-2]
        if best["value"] - inner["value"] > TIE and not best["extension"]:
            ratio = inner["lr"] / best["lr"] if k == 0 else best["lr"] / inner["lr"]
            new_lr = best["lr"] / ratio if k == 0 else best["lr"] * ratio
            return {"extend": float(f"{new_lr:.3g}"), "from": best["tree"]}
    tied = [c for c in with_lr if c["value"] >= best["value"] - TIE]
    interior = [c for c in tied if c["lr"] not in (lrs[0], lrs[-1])]
    return {"tree": max(interior or tied, key=lambda c: c["value"])["tree"]}


def plateau(runs_list, fmt_kind, pair="pair0_1", ns=FEWSHOT_N, actions_path=None, latex_path=None):
    """
    Hyperparameter selection (results_v2.tex, Setup), per model and setting, over the trees given (one per learning
    rate, named <model>_lr<x>; a tree holding the file 'grid_extension' is a grid-edge step): the rule on each tree's
    selection curve; runs that have not plateaued are extended (doubled, up to the cap), unresolved ones re-run with
    log-spaced evaluation; once every learning rate of the grid has a settled curve, choose_learning_rate decides.
    Continual choices are to be confirmed by a run at the selected budget.
    actions_path: the runs still needed, one per line: tree path|environment|run_v2.sh arguments|unit dir|flag
    (flag 'grid_extension' for a new grid-edge tree). latex_path: rows of results_v2.tex's selection table.
    """
    lines = [f"% hyperparameter selection on validation (fine 3-5), {pair}: model & setting & tree & budget & value at "
             f"budget & best & last improvement & length & status"]
    settings = [("continual", None), ("stream", None)] + [("fewshot", n) for n in ns]
    cc = pair[len("pair"):].replace("_", " ")
    actions, choices, cells = [], [], {}

    def unit_actions(tree_path, model, setting, n, epochs, lr, flag=""):
        env = f"EPOCHS={epochs}" + (" EVAL_POINTS=96" if setting == "fewshot" else "")
        env += f" LR={fmt_lr(lr)}" if lr is not None else ""
        base = os.path.join(tree_path, setting, model, pair, "seed1")
        for u in selection_units(setting, n):
            if setting == "continual":
                args = f"continual {model} \"{cc}\" 1 \"{' '.join(u[len('order'):].split('_'))}\""
            elif setting == "stream":
                args = f"stream {model} \"{cc}\" 1 {u[len('fine'):]}"
            else:
                args = f"fewshot {model} \"{cc}\" 1 {u[len('fine'):].split('_')[0]} {n}"
            actions.append(f"{tree_path}|{env}|{args}|{os.path.join(base, u)}|{flag}")

    for model, label in MODELS:
        trees = [r for r in runs_list
                 if os.path.basename(os.path.normpath(r)) == model or os.path.basename(os.path.normpath(r)).startswith(model + "_lr")]
        for setting, n in settings:
            setting_label = setting if n is None else f"fewshot N={n}"
            candidates, incomplete = [], False
            for runs in trees:
                tree = os.path.basename(os.path.normpath(runs))
                extension = os.path.exists(os.path.join(runs, GRID_EXTENSION_MARK))
                curve = validation_curve(runs, model, setting, n, pair)
                if curve is None:
                    if not extension:  # a grid-edge tree only runs the settings it was needed for
                        incomplete = True
                    continue
                r = selection_rule(curve)
                limit = cap(model, setting, n)
                r["at_cap"] = not r["plateaued"] and r["length"] >= limit
                if r["unresolved"]:
                    status = "UNRESOLVED: peak at the first evaluation, re-run with log-spaced evaluation"
                    unit_actions(runs, model, setting, n, r["length"], tree_lr(tree))
                elif r["plateaued"]:
                    status = "plateaued"
                elif r["at_cap"]:
                    status = "NOT PLATEAUED at the cap"
                else:
                    status = f"extend to {min(2 * r['length'], limit)}"
                    unit_actions(runs, model, setting, n, min(2 * r["length"], limit), tree_lr(tree))
                settled = (r["plateaued"] or r["at_cap"]) and not r["unresolved"]
                candidates.append({"tree": tree, "path": runs, "lr": tree_lr(tree), "value": r["value"],
                                   "extension": extension, "settled": settled, "r": r})
                cells[(tree, setting_label)] = r
                lines.append(emit([label, setting_label, tree, str(r["budget"]), f"{r['value']:.3f}", f"{r['best']:.3f}",
                                   str(r["last_improvement"]), str(r["length"]), status], fmt_kind))
            if not candidates:
                continue
            if incomplete or not all(c["settled"] for c in candidates):
                why = "runs missing" if incomplete else "runs to extend or re-run"
                choices.append(emit([label, setting_label, f"undecided: {why}", "-", "-"], fmt_kind))
                continue
            decision = choose_learning_rate(candidates)
            if "extend" in decision:
                src = next(c for c in candidates if c["tree"] == decision["from"])
                new_tree = f"{model}_lr{fmt_lr(decision['extend'])}"
                new_path = os.path.join(os.path.dirname(os.path.normpath(src["path"])), new_tree)
                unit_actions(new_path, model, setting, n, src["r"]["length"], decision["extend"], GRID_EXTENSION_MARK)
                choices.append(emit([label, setting_label, f"grid edge ({src['tree']}): run {new_tree}", "-", "-"], fmt_kind))
                cells[("choice", model, setting_label)] = ("edge", src["tree"])
                continue
            c = next(c for c in candidates if c["tree"] == decision["tree"])
            note = " (confirm at the budget)" if setting == "continual" else ""
            note += " (NOT PLATEAUED: at the cap)" if c["r"]["at_cap"] else ""
            note += " (at the extended edge of the grid)" if c["extension"] else ""
            choices.append(emit([label, setting_label, c["tree"] + note, str(c["r"]["budget"]), f"{c['value']:.3f}"], fmt_kind))
            cells[("choice", model, setting_label)] = ("chosen", c["tree"])
    lines += ["", "% selection: model & setting & tree (learning rate) & budget & value"] + choices
    if actions_path:
        with open(actions_path, "w") as f:
            f.write("\n".join(actions) + ("\n" if actions else ""))
        lines.append(f"% {len(actions)} runs still needed, written to {actions_path}")
    if latex_path:
        write_selection_latex(latex_path, runs_list, cells, ns)
    return lines


def write_selection_latex(path, runs_list, cells, ns):
    """
    Rows for results_v2.tex: one per model and learning rate; each cell 'budget (smoothed validation value)'.
    Bold: chosen. Marks: $^\dagger$ to be extended, $^\S$ not plateaued at the cap, $^?$ unresolved,
    $^\ddagger$ the grid is extended from here.
    """
    labels = dict(MODELS)
    cols = ["continual", "stream"] + [f"fewshot N={n}" for n in ns]
    rows = []
    for runs in runs_list:
        tree = os.path.basename(os.path.normpath(runs))
        model = tree.split("_lr")[0]
        lr = tree_lr(tree)
        out = [labels.get(model, model), "--" if lr is None else fmt_lr(lr)]
        for c in cols:
            r = cells.get((tree, c))
            if r is None:
                out.append("\\pending")
                continue
            cell = f"{r['budget']} ({r['value']:.3f})"
            if r["unresolved"]:
                cell += "$^?$"
            elif r["at_cap"]:
                cell += "$^\\S$"
            elif not r["plateaued"]:
                cell += "$^\\dagger$"
            choice = cells.get(("choice", model, c))
            if choice and choice[1] == tree:
                cell = f"\\textbf{{{cell}}}" + ("$^\\ddagger$" if choice[0] == "edge" else "")
            out.append(cell)
        rows.append("    " + " & ".join(out) + " \\\\")
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


TABLES = {"continual": table_continual, "starting": table_starting, "pairs": table_pairs, "stream": table_stream,
          "fewshot": table_fewshot, "e40": table_e40}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["runs_v2"],
                   help="Result trees. The tables read the first; --plateau compares all (one per learning rate).")
    p.add_argument("--tables", nargs="*", default=list(TABLES), choices=list(TABLES))
    p.add_argument("--part", choices=["test", "val"], default="test")
    p.add_argument("--format", choices=["latex", "md"], default="latex")
    p.add_argument("--plateau", action="store_true")
    p.add_argument("--plateau-pair", default="pair0_1")
    p.add_argument("--actions", default=None, help="--plateau: write the runs still needed (tree path|env|run_v2.sh args|unit dir).")
    p.add_argument("--selection-latex", default=None, help="--plateau: write the selection table rows for results_v2.tex.")
    p.add_argument("--partial", action="store_true",
                   help="Single-stream and few-shot: average over the fine classes that have run (cells marked *), "
                        "e.g. for the pilot on fine class 3 only.")
    args = p.parse_args()
    global PARTIAL
    PARTIAL = args.partial
    for runs in args.runs:
        if not os.path.isdir(runs):
            sys.exit(f"no such directory: {runs}")
    warnings.simplefilter("always")
    for name in args.tables:
        for line in TABLES[name](args.runs[0], args.part, args.format):
            print(line)
        print()
    if args.plateau:
        for line in plateau(args.runs, args.format, args.plateau_pair, actions_path=args.actions,
                            latex_path=args.selection_latex):
            print(line)


if __name__ == "__main__":
    main()
