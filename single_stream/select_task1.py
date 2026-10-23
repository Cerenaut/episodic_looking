"""
Apply the task-1 selection rule (single_stream_paper plan.md section 6, step 2) to a round of c = 1 runs.

Step size: highest validation accuracy at the end of the last epoch (flags a grid-edge winner).
Epochs: smallest whole number of epochs whose end-of-epoch accuracy is within `--tol` of that run's best.

Usage: python -m single_stream.select_task1 runs_ss/task1_r1_select_20261024 [--tol 0.002]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def end_of_epoch_acc(run_dir):
    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    with open(os.path.join(run_dir, "checkpoints.jsonl")) as f:
        records = [json.loads(line) for line in f]
    epoch_len = records[-1]["t"] // args["epochs"]
    by_t = {r["t"]: r["acc"] for r in records}
    return args, [by_t.get(epoch_len * (e + 1)) for e in range(args["epochs"])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("round_dir")
    p.add_argument("--tol", type=float, default=0.002)
    a = p.parse_args()

    variants = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(a.round_dir, "*_c1_lr*_s0"))):
        if not os.path.exists(os.path.join(run, "summary.json")):
            print(f"incomplete: {run}")
            continue
        args, accs = end_of_epoch_acc(run)
        tag = re.match(r"(.+)_c1_lr", os.path.basename(run)).group(1)
        variants[tag].append((args["lr"], accs))

    for tag, runs in variants.items():
        runs.sort()
        print(f"\n{tag}: end-of-epoch validation accuracy")
        for lr, accs in runs:
            print(f"  lr {lr:<7g} " + "  ".join(f"{x:.4f}" for x in accs))
        best_i = max(range(len(runs)), key=lambda i: runs[i][1][-1])
        lr, accs = runs[best_i]
        edge = " (GRID EDGE: extend the grid)" if best_i in (0, len(runs) - 1) else ""
        best = max(accs)
        epochs = next(e + 1 for e, x in enumerate(accs) if x >= best - a.tol)
        print(f"  selected lr {lr:g}{edge}; epochs {epochs} (best end-of-epoch {best:.4f}, tol {a.tol})")


if __name__ == "__main__":
    main()
