"""Stop-rule check (Dave, 14 Oct 2026): trained-class accuracy of every finished few-shot unit under the given trees,
first evaluation, best and last (part val for selection runs, whose test numbers are not opened; test for final runs).
Flags: N > 1 with the last value below 0.60; N = 1 summarised per model (consistently below 0.60 is the warning).
Usage (repo root): python analysis/fewshot_rerun/check.py --part val runs_v2_pilot_fewshot/stm_lr0.01 ..."""
import argparse, glob, os, re
from collections import defaultdict
p = argparse.ArgumentParser(); p.add_argument("--part", choices=["val", "test"], required=True); p.add_argument("trees", nargs="+")
p.add_argument("--seeds", default=None, help="comma list: only these seeds"); a = p.parse_args()
seeds = None if a.seeds is None else {int(s) for s in a.seeds.split(",")}
rows, n1 = [], defaultdict(list)
for tree in a.trees:
    for unit in sorted(glob.glob(os.path.join(tree, "fewshot", "*", "pair*", "seed*", "fine*_n*"))):
        if not os.path.exists(os.path.join(unit, "job.done")):
            continue
        m = re.search(r"fewshot/([^/]+)/(pair[^/]+)/seed(\d+)/fine(\d)_n(\d+)$", unit)
        model, pair, seed, c, n = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        if seeds is not None and seed not in seeds:
            continue
        files = [f for f in glob.glob(os.path.join(unit, "**", "results_*.txt"), recursive=True)
                 if ("pretrain" not in os.path.basename(f) and "images" not in f)
                 and (f.endswith("_val.txt") == (a.part == "val"))]
        if len(files) != 1:
            print("?", unit, files); continue
        pts = [(int(e), float(v)) for line in open(files[0])
               for (cls, mode, e, v) in [re.match(r"\[.*?\], \[(.*?)\], (\w+), (\d+), ([\d.]+)", line).groups()]
               if mode == "evaluate" and cls.strip("'") == c]
        pts.sort(); first, last, best = pts[0][1], pts[-1][1], max(v for _, v in pts)
        flag = "LOW" if n > 1 and last < 0.60 else ""
        rows.append((model, n, c, seed, first, best, last, pts[-1][0] + 1, flag))
        if n == 1: n1[model].append(last)
print(f"part {a.part}: model N fine seed | first best last (last epoch) flag")
for r in rows: print(f"{r[0]:9s} {r[1]:4d} {r[2]} s{r[3]:<2d} | {r[4]:.3f} {r[5]:.3f} {r[6]:.3f} ({r[7]}) {r[8]}")
for m, v in n1.items():
    print(f"N=1 {m}: {len(v)} units, last mean {sum(v)/len(v):.3f}, below 0.60 in {sum(x < 0.6 for x in v)}")
print(f"N>1 below 0.60 at the last evaluation: {sum(1 for r in rows if r[8])} of {sum(1 for r in rows if r[1] > 1)}")
