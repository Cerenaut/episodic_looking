"""RL STM few-shot final runs (runs_v2_fs_rl/rl_lr0.1; budgets N=1/4/16/64/400 = 1/24/26/21/21 RL epochs), test part,
last evaluation: per N, trained-class accuracy (fine 3-5 units, mean over seeds then classes; sd over per-seed means of
the three classes), fine 1-2 retention, and the four-set mean (sets 12, 3, 4, 5); against the draft's RL row
(next_draft_experiments.tex tab:fewshot-final, trained .484/.608/.703/.793/.761). Plot of trained class per N.
Usage (repo root): python analysis/fewshot_rerun/rl_summary.py"""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "runs_local/fsfinal_rl_20261015"; TREE = "runs_v2_fs_rl/rl_lr0.1/fewshot/rl/pair0_1"
DRAFT = {1: .484, 4: .608, 16: .703, 64: .793, 400: .761}
def last(unit):
    f = [f for f in glob.glob(f"{unit}/**/results_*.txt", recursive=True)
         if not f.endswith("_val.txt") and "pretrain" not in os.path.basename(f) and "images" not in f]
    assert len(f) == 1, (unit, f)
    d = {}
    for l in open(f[0]):
        cl, m, e, v = re.match(r"\[.*?\], \[(.*?)\], (\w+), (\d+), ([\d.]+)", l).groups()
        if m == "evaluate": d[cl.strip("'")] = (int(e), float(v))  # file is in epoch order; keep the last
    return {k: v[1] for k, v in d.items()}
rows = {}
print("N  seeds | trained mean (sd over seeds) | fine3 fine4 fine5 | fine1-2 | four-set | draft trained")
for n in (1, 4, 16, 64, 400):
    seeds = sorted(int(re.search(r"seed(\d+)", p).group(1)) for p in glob.glob(f"{TREE}/seed*") if os.path.isdir(f"{p}/fine3_n{n}"))
    tr = np.array([[last(f"{TREE}/seed{s}/fine{c}_n{n}")[c] for c in "345"] for s in seeds])
    f12 = np.array([[last(f"{TREE}/seed{s}/fine{c}_n{n}")["12"] for c in "345"] for s in seeds])
    four = np.array([[np.mean(list(last(f"{TREE}/seed{s}/fine{c}_n{n}").values())) for c in "345"] for s in seeds])
    per_seed = tr.mean(1); rows[n] = (per_seed.mean(), per_seed.std(ddof=1), tr)
    print(f"{n:<3d} {len(seeds):>3d}   | {per_seed.mean():.3f} ({per_seed.std(ddof=1):.3f})            | "
          f"{tr[:,0].mean():.3f} {tr[:,1].mean():.3f} {tr[:,2].mean():.3f} | {f12.mean():.3f}   | {four.mean():.3f}    | {DRAFT[n]:.3f}")
ns = list(rows); x = np.arange(len(ns))
plt.figure(figsize=(7, 4))
plt.errorbar(x, [rows[n][0] for n in ns], [rows[n][1] for n in ns], fmt="o-", capsize=4, label="RL STM, rerun (mean, sd over seeds)")
for n, xi in zip(ns, x):
    plt.scatter(np.full(rows[n][2].size, xi) + np.tile([-.12, 0, .12], len(rows[n][2])), rows[n][2].ravel(), s=8, alpha=.4,
                c=np.tile(["C0", "C1", "C2"], len(rows[n][2])))
plt.plot(x, [DRAFT[n] for n in ns], "s--", color="grey", label="RL STM, draft")
plt.xticks(x, [f"N={n}" for n in ns]); plt.ylabel("trained fine-class test acc"); plt.axhline(.6, color="k", lw=.5)
plt.title("Few-shot RL STM, pair A (dots: fine 3/4/5 per seed)"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(f"{OUT}/rl_fewshot_test.png", dpi=120); print(f"{OUT}/rl_fewshot_test.png")
