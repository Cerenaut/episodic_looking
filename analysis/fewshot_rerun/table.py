"""Few-shot final results, pair A, test, last evaluation: all models (RL STM rerun from runs_v2_fs_rl, the rest from
runs_v2; combined through the symlinks in tree/fewshot). Per model and N: trained-class mean over seeds then fine
classes (as metrics_v2.table_fewshot), sd over per-seed means of fine 3-5, seed count, fine 1-2. Plot of trained class.
Usage (repo root): python analysis/fewshot_rerun/table.py"""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = "runs_local/fewshot_final_20261015"; T = f"{O}/tree/fewshot"
MODELS = [("rl", "CLS/STM, RL"), ("stm", "Supervised STM head"), ("linear", "Linear probe"), ("ncm", "NCM"),
          ("flymodel", "FlyModel"), ("sdmlp", "SDMLP"), ("ltm", "LTM-only fine-tuning")]
NS = (1, 4, 16, 64, 400)
def last(unit):
    f = [f for f in glob.glob(f"{unit}/**/results_*.txt", recursive=True)
         if not f.endswith("_val.txt") and "pretrain" not in os.path.basename(f) and "images" not in f]
    assert len(f) == 1, (unit, f)
    d = {}
    for l in open(f[0]):
        m = re.match(r"\[.*?\], \[(.*?)\], (\w+), (\d+), ([\d.]+)", l)
        if m and m.group(2) == "evaluate": d[m.group(1).strip("'")] = float(m.group(4))
    return d
res = {}
lines = ["| Model | " + " | ".join(f"N={n}" for n in NS) + " | fine 1-2 at N=1 / 400 |", "|---" * (len(NS) + 2) + "|"]
for m, label in MODELS:
    cells, f12s = [], []
    for n in NS:
        seeds = sorted(glob.glob(f"{T}/{m}/pair0_1/seed*/fine3_n{n}/job.done"))
        seeds = [re.search(r"seed(\d+)", s).group(1) for s in seeds]
        tr = np.array([[last(f"{T}/{m}/pair0_1/seed{s}/fine{c}_n{n}")[c] for c in "345"] for s in seeds])
        f12 = np.mean([[last(f"{T}/{m}/pair0_1/seed{s}/fine{c}_n{n}")["12"] for c in "345"] for s in seeds])
        ps = tr.mean(1); res[(m, n)] = (tr.mean(0).mean(), ps.std(ddof=1)); f12s.append(f12)
        cells.append(f"{tr.mean(0).mean():.3f} ± {ps.std(ddof=1):.3f} ({len(seeds)})")
    lines.append(f"| {label} | " + " | ".join(cells) + f" | {f12s[0]:.3f} / {f12s[-1]:.3f} |")
open(f"{O}/table_test.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
x = np.arange(len(NS)); plt.figure(figsize=(8, 4.5))
for i, (m, label) in enumerate(MODELS):
    plt.errorbar(x + (i - 3) * 0.04, [res[(m, n)][0] for n in NS], [res[(m, n)][1] for n in NS], fmt="o-", capsize=3,
                 label=label, lw=2.2 if m == "rl" else 1, color="k" if m == "rl" else None)
plt.xticks(x, [f"N={n}" for n in NS]); plt.ylabel("trained fine-class test acc"); plt.legend(fontsize=7)
plt.title("Few-shot, pair A (mean over fine 3-5; bars: sd over seeds; 10 seeds at N=1, 4, 5 above)")
plt.tight_layout(); plt.savefig(f"{O}/fewshot_test.png", dpi=120); print(f"{O}/fewshot_test.png")
