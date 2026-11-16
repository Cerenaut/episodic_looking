# Round rl24cd (18 Oct 2026): RL STM continual at 24 epochs per phase on pairs C, D (seed 1) against the 12-epoch
# finals (3 seeds). Full-run metrics, plus the trained class's accuracy at epoch 6/12/18/24 within each phase of the
# 24-epoch run (later phases start from the state the earlier 24-epoch phases left).
# Usage: python analysis/rl24cd.py [test|val]
import os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import metrics_v2 as mv
from plot_comparison import cl_metrics, r_matrix
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
TREES = {"e12 finals": "runs_local/continual_final_20261015/tree", "e24": "runs_v2_e24cd"}
KEYS = ["ACC", "LA", "forgetting_eop", "BWT", "BWT_0", "FWT"]
for letter, pair in [("C", "pair5_6"), ("D", "pair15_16")]:
    for name, tree in TREES.items():
        per_seed = {}
        for seed, seed_dir in mv.seeds_of(os.path.join(tree, "continual", "rl", pair)):
            for order in mv.ORDERS:
                unit = os.path.join(seed_dir, "order" + "_".join(map(str, order)))
                if not mv.finished(unit):
                    continue
                piv = mv.load_run_pivot(mv.one_file(unit, SPLIT))
                base = mv.baseline_series(tree, "", "rl", pair, seed, unit, SPLIT)
                piv.attrs["baseline"], piv.attrs["baseline_source"] = base, "file"
                m = cl_metrics(r_matrix(piv, order), order)
                E = len(piv.index) // 3
                for k in (6, 12, 18, 24):
                    if k <= E:
                        m[f"LA@{k}"] = np.mean([piv[str(c)].iloc[t * E + k - 1] for t, c in enumerate(order)])
                per_seed.setdefault(seed, []).append(m)
        per_seed = {s: v for s, v in per_seed.items() if len(v) == 3}
        if not per_seed:
            print(f"{letter} {name}: no finished seeds"); continue
        keys = KEYS + [k for k in per_seed[min(per_seed)][0] if k.startswith("LA@")]
        def fmt(ms):
            return " ".join(f"{k} {np.mean([m[k] for m in ms]):.3f}" for k in keys)
        means = {k: [np.mean([m[k] for m in v]) for v in per_seed.values()] for k in keys}
        print(f"{letter} {name:10s} n={len(per_seed)} " + " ".join(f"{k} {np.mean(v):.4f}" for k, v in means.items()))
        if name.startswith("e12") and 1 in per_seed:
            print(f"{letter} {'e12 seed1':10s} n=1 " + fmt(per_seed[1]))
