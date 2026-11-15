"""
LaTeX tables of the head tuning stages 1 and 1b (Dave 15 Nov 2026; comparisons/queue_tune1.sh): every setting, reset
and kept, on lp and dih c 512, seed 0, task 2 to 2,048 exposures; the core metrics (single_stream/score.py) per family
and the mean overall score. One table per head, rows sorted by the mean overall score.

    python -m single_stream.tune_tables --root runs_ss/mnist_clustered_single_stream --out <file.tex>
"""
import argparse
import os
import re

import numpy as np

from single_stream.score import FOUR_TASK, four_task

STAGES = ("comparisons/tune1", "comparisons/tune1b")
HEADS = (("knn", "kNN"), ("fly", "FlyModel"), ("sdmlp", "SDMLP"))
RUN = re.compile(r"(.+)_(reset|keep)_(lp|dih)_c512_s0$")


def collect(root):
    """{(setting, arm): {fam: scores}} over the stage 1 and 1b runs, with the stage each setting came from."""
    out, stage = {}, {}
    for d in STAGES:
        for name in sorted(os.listdir(os.path.join(root, d))):
            m = RUN.match(name)
            if m and os.path.exists(os.path.join(root, d, name, "summary.json")):
                out.setdefault((m[1], m[2]), {})[m[3]] = four_task(os.path.join(root, d, name))
                stage[m[1]] = "1b" if d.endswith("1b") else "1"
    return out, stage


def table(rows, stage, head, label):
    keys = [name for _, name in FOUR_TASK]
    lines = [r"\begin{table}[H]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llccccccccc}", r"\toprule",
             r" & & & \multicolumn{4}{c}{label permutation} & \multicolumn{4}{c}{dihedral} \\",
             r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
             r"Setting & Arm & Mean & M & G & P & Overall & M & G & P & Overall \\", r"\midrule"]
    items = [(k, v) for k, v in rows.items() if k[0].split("_")[0] == head and len(v) == 2]
    items.sort(key=lambda kv: -np.mean([kv[1][f]["Score"] for f in ("lp", "dih")]))
    for (setting, arm), v in items:
        cells = [f"{np.mean([v[f]['Score'] for f in ('lp', 'dih')]):.3f}"]
        for f in ("lp", "dih"):
            cells += [f"{v[f][k]:.2f}" for k in keys] + [f"{v[f]['Score']:.3f}"]
        name = setting.replace("_", " ") + (r"$^\dagger$" if stage[setting] == "1b" else "")
        lines.append(f"{name} & {'reset' if arm == 'reset' else 'kept'} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{label} head, stages 1 and 1b ($^\dagger$: 1b), c 512, seed 0, task 2 to 2,048 exposures; "
              r"sorted by the mean overall score over the two families. M, G, P: Memorisation, Generalisation, "
              r"Protection.}", rf"\label{{tab:{head}}}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, stage = collect(args.root)
    with open(args.out, "w") as f:
        f.write("\n".join(table(rows, stage, h, label) for h, label in HEADS))
    print(args.out)


if __name__ == "__main__":
    main()
