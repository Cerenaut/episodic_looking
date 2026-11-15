"""
Overall score (Dave 13 Nov 2026): equal weight to each core metric of an experiment, reported alongside the
breakdown (never instead of it). The aim is to map each model's range of behaviours and trade-offs, not to maximise
the headline number.

Four-task experiment (run_tasks; label permutation and dihedral scored separately), over tasks 2-4 (task 1 is the
LTM's pre-training), every class, at a class's n-th exposure in its task, n in POINTS:
  Memorisation    recall on the class's training samples seen so far
  Generalisation  recall on held-out samples of the class
  Protection      accuracy on held-out samples of the current task's other classes
  Score           mean of the three
  Worst           their minimum ("Overall - worst", Dave 16 Nov 2026: favours a model robust across the metrics over
                  one making a hard trade-off); over several runs, the minimum of the mean metrics, not the mean of
                  per-run minima
The held-out-digit experiment adds Resilience, Retention and Protection-2 (definitions in plot_final.py); its score
is the mean of its six metrics (to be added with that experiment's port to the current models).

    python -m single_stream.score <run dir> [<run dir> ...]     # one row per run, then the mean
"""
import argparse
import json
import os

import numpy as np

POINTS = (1, 2, 4, 8, 16, 32, 64)
FOUR_TASK = (("memorisation", "Memorisation"), ("recall", "Generalisation"), ("protection", "Protection"))


def four_task(run, points=POINTS, first_task=1, last_task=None, max_t_task=None):
    """{metric: value, "Score": mean} for one run_tasks run; metrics averaged over (task, class, n) records. last_task
    and max_t_task restrict the window (e.g. task 2's first 2,048 exposures, to compare four-task runs with the
    two-task tuning runs)."""
    rows = [json.loads(line) for line in open(os.path.join(run, "fewshot.jsonl"))]
    rows = [r for r in rows if r["task"] >= first_task and r["threshold"] in points
            and (last_task is None or r["task"] <= last_task) and (max_t_task is None or r["t_task"] <= max_t_task)]
    if not rows:
        raise ValueError(f"{run}: no few-shot records in tasks >= {first_task} at {points}")
    out = {name: float(np.mean([r[key] for r in rows])) for key, name in FOUR_TASK}
    out["Score"] = float(np.mean([out[name] for _, name in FOUR_TASK]))
    out["Worst"] = float(np.min([out[name] for _, name in FOUR_TASK]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    names = [name for _, name in FOUR_TASK] + ["Score", "Worst"]
    print(f"{'run':50s} " + " ".join(f"{n[:6]:>7s}" for n in names))
    scores = []
    for run in args.runs:
        s = four_task(run)
        scores.append([s[n] for n in names])
        print(f"{run[-50:]:50s} " + " ".join(f"{v:7.3f}" for v in scores[-1]))
    if len(scores) > 1:
        mean = np.mean(scores, 0)
        mean[-1] = mean[:len(FOUR_TASK)].min()
        print(f"{'mean':50s} " + " ".join(f"{v:7.3f}" for v in mean))


if __name__ == "__main__":
    main()
