"""Fine 1-2 (pretraining classes, set '12') validation accuracy during the few-shot selection runs (pair A, seed 1):
first, min, last per unit, RL STM vs supervised STM head, and a curve plot. Validation only.
Usage (repo root): python analysis/fewshot_rerun/sel_fine12.py"""
import glob, os, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "runs_local/fsel_rl_curves_20261014"
TREES = {"RL STM": "runs_v2_pilot_fewshot/rl_lr0.1/fewshot/rl", "Sup. STM head": "runs_v2_pilot_fewshot/stm_lr0.01/fewshot/stm"}
def curve(unit, s):
    f = [f for f in glob.glob(f"{unit}/**/results_*_val.txt", recursive=True) if "pretrain" not in os.path.basename(f) and "images" not in f][0]
    pts = sorted((int(e), float(v)) for l in open(f) for (cl, m, e, v) in [re.match(r"\[.*?\], \[(.*?)\], (\w+), (\d+), ([\d.]+)", l).groups()]
                 if m == "evaluate" and cl.strip("'") == s)
    L = pts[-1][0] + 1
    return [(e + 1) / L for e, _ in pts], [v for _, v in pts]
fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
print("model         N fine | fine12 first  min   last | trained last")
for i, n in enumerate((1, 4, 16, 64, 400)):
    for (name, tree), ls in zip(TREES.items(), ("-", "--")):
        for c, col in zip("345", ("C0", "C1", "C2")):
            u = f"{tree}/pair0_1/seed1/fine{c}_n{n}"
            x, y = curve(u, "12"); _, t = curve(u, c)
            print(f"{name:13s} {n:3d}  {c}   |   {y[0]:.3f}  {min(y):.3f} {y[-1]:.3f} |  {t[-1]:.3f}")
            ax[i].plot(x, y, ls, color=col, label=f"{name}, training fine {c}")
    ax[i].set_title(f"N={n}"); ax[i].set_xlabel("fraction of run"); ax[i].legend(fontsize=6)
ax[0].set_ylabel("fine 1-2 val acc (no forgetting = start)")
plt.tight_layout(); plt.savefig(f"{OUT}/fine12_seed1.png", dpi=110); print(f"{OUT}/fine12_seed1.png")
