"""RL STM few-shot selection, 24- vs 48-epoch rounds (pair A, seed 1, validation): per N, the four-set mean curve
(mean over fine 3-5 units of the mean of sets 12, 3, 4, 5; the selection rule's curve), the trained class and fine 1-2,
x in epochs. The two rounds use different STM pre-trainings (each pod pre-trains its own seed-1 STM).
Usage (repo root): python analysis/fewshot_rerun/sel_e24_vs_e48.py"""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "runs_local/fsel_rl_curves_20261014"
def tree(r, n): return "runs_v2_pilot_fewshot/rl_lr0.1" if r == 24 else f"runs_v2_pilot_fewshot_e48_n{n}/rl_lr0.1"
def sets(unit):
    f = [f for f in glob.glob(f"{unit}/**/results_*_val.txt", recursive=True) if "pretrain" not in os.path.basename(f) and "images" not in f][0]
    d = {}
    for l in open(f):
        cl, m, e, v = re.match(r"\[.*?\], \[(.*?)\], (\w+), (\d+), ([\d.]+)", l).groups()
        if m == "evaluate": d.setdefault(cl.strip("'"), {})[int(e)] = float(v)
    E = sorted(d["12"]); return np.array(E) + 1, {s: np.array([d[s][e] for e in E]) for s in ("12", "3", "4", "5")}
fig, ax = plt.subplots(3, 4, figsize=(18, 10), sharex="col")
for j, n in enumerate((4, 16, 64, 400)):
    for r, ls in ((24, "--"), (48, "-")):
        means = []
        for c, col in zip("345", ("C0", "C1", "C2")):
            x, s = sets(f"{tree(r, n)}/fewshot/rl/pair0_1/seed1/fine{c}_n{n}")
            means.append(np.mean([s[k] for k in s], 0))
            ax[1, j].plot(x, s[c], ls, color=col, label=f"{r}-ep round, fine {c}")
            ax[2, j].plot(x, s["12"], ls, color=col, label=f"{r}-ep round, training fine {c}")
        ax[0, j].plot(x, np.mean(means, 0), ls, color="k", label=f"{r}-ep round")
    ax[0, j].set_title(f"N={n}"); ax[2, j].set_xlabel("RL epoch (4,000 steps x 16 envs)")
    for i in range(3): ax[i, j].legend(fontsize=6); ax[i, j].axvline(24, color="grey", lw=0.5)
ax[0, 0].set_ylabel("four-set val mean (selection curve)"); ax[1, 0].set_ylabel("trained class val acc"); ax[2, 0].set_ylabel("fine 1-2 val acc")
plt.tight_layout(); plt.savefig(f"{OUT}/e24_vs_e48.png", dpi=100); print(f"{OUT}/e24_vs_e48.png")
