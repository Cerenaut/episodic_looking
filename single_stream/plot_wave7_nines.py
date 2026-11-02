"""
Wave 7 (Dave, 2 Nov 2026): per training nine, the best p(9) reached by sample 200 (policy mean, seed mean) per config,
next to the reachable optimum of each bias (debug_stuck_nine); and the samples to reach memorised / held-out 0.5.

  python -m single_stream.plot_wave7_nines --run runs_ss/stm_frozen_20261101/snap9 --debug runs_ss/stm_frozen_20261101/debug_nine2100 --out <dir>
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_stream.analyse_stm_diag import load, presentations
from single_stream.plot_stm_frozen import WAVE7, load_stm, load_ltm_any, SURFACE, INK, INK2

CONFIGS = [(f"{row.split(',')[-1].strip()}: {label}", names) for row, groups in WAVE7 if "1000" not in row
           for label, names in groups]


def best_p9(run, name, max_sample=200):
    best = {}
    for p in presentations(load(run, name), max_sample):
        best[p["index"]] = max(best.get(p["index"], 0.0), p["post"])
    return best


def first_reach(rows, k, level=0.5):
    return next((r[0] for r in rows if r[k] is not None and r[k] >= level), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--debug", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reach = {m: json.load(open(os.path.join(args.debug, *([] if m == "where" else ["what"]), "reachable.json")))["images"]
             for m in ("where", "what")}
    nines = sorted(reach["where"], key=lambda j: reach["where"][j]["p_y0"])
    cols = [("unbiased", [reach["where"][j]["p_y0"] for j in nines]),
            ("optimal where bias", [reach["where"][j]["p_y_opt"] for j in nines]),
            ("optimal what bias", [reach["what"][j]["p_y_opt"] for j in nines])]
    for label, names in CONFIGS:
        bests = [best_p9(args.run, n) for n in names]
        cols.append((label, [np.mean([b.get(int(j), np.nan) for b in bests]) for j in nines]))
    m = np.array([c[1] for c in cols]).T
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=SURFACE)
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if m[i, j] < 0.55 else INK)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(nines)), [f"nine {j}" for j in nines], fontsize=8.5)
    ax.axvline(2.5, color="white", lw=3)
    ax.set_title("p(9) per training nine: unbiased, reachable optimum per bias, and the best reached by the RL STM by "
                 "sample 200 (seed mean)", color=INK, loc="left", fontsize=10.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="p(9)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "wave7_nines.png"), dpi=150, facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(args.out, "wave7_reach.md"), "w") as f:
        f.write("| config | samples to memorised >= 0.5 (per seed) | samples to held-out >= 0.5 (per seed) |\n|---|---|---|\n")
        ltm = load_ltm_any(args.run, "ltm200_lr0.001")
        f.write(f"| LTM only | {first_reach(ltm, 1)} | {first_reach(ltm, 2)} |\n")
        for label, names in CONFIGS:
            rs = [load_stm(args.run, n)[1] for n in names]
            f.write(f"| {label} | {', '.join(str(first_reach(r, 1)) for r in rs)} | "
                    f"{', '.join(str(first_reach(r, 2)) for r in rs)} |\n")
    print(open(os.path.join(args.out, "wave7_reach.md")).read())


if __name__ == "__main__":
    main()
