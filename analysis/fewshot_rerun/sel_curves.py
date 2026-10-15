"""Trained-class validation curves of the few-shot selection runs (pair A, seed 1): RL STM (24 epochs, lr 0.1) vs the
supervised STM head (lr 0.01), one panel per N, x as fraction of each run's length (units differ: RL epoch = 4,000
steps x 16 envs; head epoch = one pass over 2N images). Validation only.
Usage (repo root): python analysis/fewshot_rerun/sel_curves.py"""
import glob, os, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "runs_local/fsel_rl_curves_20261014"
TREES = {"RL STM": "runs_v2_pilot_fewshot/rl_lr0.1/fewshot/rl", "Sup. STM head": "runs_v2_pilot_fewshot/stm_lr0.01/fewshot/stm"}
def curve(unit, c):
    f = [f for f in glob.glob(f"{unit}/**/results_*_val.txt", recursive=True) if "pretrain" not in os.path.basename(f) and "images" not in f][0]
    pts = sorted((int(e), float(v)) for l in open(f) for (cl, m, e, v) in [re.match(r"\[.*?\], \[(.*?)\], (\w+), (\d+), ([\d.]+)", l).groups()]
                 if m == "evaluate" and cl.strip("'") == c)
    L = pts[-1][0] + 1
    return [(e + 1) / L for e, _ in pts], [v for _, v in pts], L
fig, ax = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for i, n in enumerate((1, 4, 16, 64, 400)):
    for (name, tree), ls in zip(TREES.items(), ("-", "--")):
        for c, col in zip("345", ("C0", "C1", "C2")):
            x, y, L = curve(f"{tree}/pair0_1/seed1/fine{c}_n{n}", c)
            ax[i].plot(x, y, ls, color=col, label=f"{name} fine {c} ({L} ep)")
    ax[i].axhline(0.6, color="k", lw=0.5); ax[i].set_title(f"N={n}"); ax[i].set_xlabel("fraction of run"); ax[i].legend(fontsize=6)
ax[0].set_ylabel("trained-class val acc")
plt.tight_layout(); plt.savefig(f"{OUT}/curves_seed1.png", dpi=110); print(f"{OUT}/curves_seed1.png")
