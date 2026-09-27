"""
Diagnostic for the SDMLP comparison: how much of each continual phase's change in the actor output is carried
by parameters shared by every clique (output bias, input LayerNorm affine) rather than by the active clique,
plus cell usage and within-episode mask stability. Reuses analyze_stm_inspectability.py; no training.
"""
import sys, types
import numpy as np
import torch

sys.path.insert(0, "/Users/gideon/Dev/episodic_looking")
import analyze_stm_inspectability as A

PRE = "../cifar_100_pretrain/variants/stm_rl_fixed_e13_pt12.pth"
PHASES = [PRE] + [f"../cifar_100_pretrain/variants/stm_rl_fixed_e13_pt12_c345_phase{p}.pth" for p in (3, 4, 5)]
LTM = "../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth"

args = types.SimpleNamespace(ltm_checkpoint=LTM, stm_checkpoint=PRE, data_path="../cifar-100-python",
                             coarse_classes=[0, 1], fine_groups=[1, 2, 3, 4, 5], split="test",
                             max_per_label=40, sparsity=32, actor_training="rl")
dev = A.get_device()
rng = np.random.default_rng(0)
images, yc, yf, groups = A.load_images(args, rng)

def load(path):
    a = types.SimpleNamespace(**{**vars(args), "stm_checkpoint": path})
    return A.build_model(a, dev)

def actor_inputs(model):
    """Record the actor input x at every step of the mean-bias episode."""
    probe = A.StmProbe(model)
    xs = []
    orig = probe.actor
    def cap(x):
        xs.append(x.detach().clone())
        return orig(x)
    probe.actor = cap
    A.run_episodes(model, probe, images, steps=8, batch_size=250, device=dev, record=True)
    return torch.cat(xs, 0)  # [N*T, D]

@torch.no_grad()
def decompose(model, x):
    cont = model.model_actor
    mlp = cont.get_model(0)
    idx = cont.encode(x)
    mask = mlp.create_active_mask(x, active_indices=idx)
    out, hidden = A.mlp_forward_capture(mlp, x, mask)
    b3 = mlp.layers[-1].bias
    return out, out - b3, b3, idx

models = [load(p) for p in PHASES]
x = actor_inputs(models[0])  # same inputs for all checkpoints, so differences are pure weight effects
print(f"actor inputs: {tuple(x.shape)}")

outs = [decompose(m, x) for m in models]
names = ["pre", "ph3", "ph4", "ph5"]

# Cell usage and within-episode mask stability (pre-trained model)
idx0 = outs[0][3].cpu().numpy()
usage = np.bincount(idx0.ravel(), minlength=1000)
print(f"cells never active: {(usage == 0).sum()} / 1000;  cells carrying 50% of activations: "
      f"{np.searchsorted(np.cumsum(np.sort(usage)[::-1]) / usage.sum(), 0.5) + 1}")

# Composition of the actor output logits in the pre-trained model
out, clique, b3, _ = outs[0]
print(f"\n[pre] ||b3||={b3.norm():.3f}  mean||W3h||={clique.norm(dim=1).mean():.3f}  "
      f"||mean over images of W3h||={clique.mean(0).norm():.3f}")

print("\nChange from previous checkpoint on fixed inputs (actor pre-activation, 128-d):")
print("step      mean||dOut||  ||d b3||  mean||d clique||  ||mean_img dOut|| (common part)  common/mean")
for i in range(1, 4):
    d_out = outs[i][0] - outs[i - 1][0]
    d_b3 = outs[i][2] - outs[i - 1][2]
    d_cl = outs[i][1] - outs[i - 1][1]
    common = d_out.mean(0).norm()
    print(f"{names[i-1]}->{names[i]}  {d_out.norm(dim=1).mean():10.4f}  {d_b3.norm():8.4f}  {d_cl.norm(dim=1).mean():14.4f}"
          f"  {common:14.4f}  {common / d_out.norm(dim=1).mean():10.2f}")

# LayerNorm affine drift
for i in range(1, 4):
    ln0, ln1 = models[i - 1].model_actor.get_model(0).ln, models[i].model_actor.get_model(0).ln
    print(f"LN {names[i-1]}->{names[i]}: ||d gamma||={(ln1.weight-ln0.weight).norm():.4f} "
          f"(||gamma||={ln0.weight.norm():.2f}), ||d beta||={(ln1.bias-ln0.bias).norm():.4f}")

# Fraction of hidden cells whose incoming weights changed per phase
for i in range(1, 4):
    W0, W1 = models[i - 1].model_actor.get_model(0).layers[0].weight, models[i].model_actor.get_model(0).layers[0].weight
    changed = ((W1 - W0).abs().sum(1) > 1e-7).float().mean()
    print(f"layer-1 rows changed {names[i-1]}->{names[i]}: {changed:.2%}")

# Saturation of the actor output after each phase (clamp at +-10, then the policy mean)
for i, (o, _, _, _) in enumerate(outs):
    print(f"{names[i]}: |pre-act| median {o.abs().median():.2f}, frac |pre-act|>=3 (tanh saturated): {(o.abs() >= 3).float().mean():.2%}, "
          f"between-image sd / |mean| per dim: {(o.std(0) / o.mean(0).abs().clamp_min(1e-6)).median():.3f}")

# Row layout: batches of 250 images, each contributing 8 consecutive step blocks
Ntot = len(images); rows = {}
r = 0
for start in range(0, Ntot, 250):
    B = min(250, Ntot - start)
    for t in range(8):
        for j in range(B):
            rows[(start + j, t)] = r + j
        r += B
def cliques(i_model, t):
    idx = outs[i_model][3].cpu().numpy()
    return np.stack([idx[rows[(n, t)]] for n in range(Ntot)])
def overlap(a, b):
    return np.array([len(np.intersect1d(x, y)) for x, y in zip(a, b)])
C7 = cliques(0, 7); C0 = cliques(0, 0)
perm = rng.permutation(Ntot)
diff_cls = yc != yc[perm]
print(f"\nclique overlap (k=32, chance 1.02): random pairs step 7: {overlap(C7, C7[perm]).mean():.1f} "
      f"(different coarse class {overlap(C7[diff_cls], C7[perm][diff_cls]).mean():.1f}); "
      f"same image step 0 vs 7: {overlap(C0, C7).mean():.1f}; "
      f"step 6 vs 7: {overlap(cliques(0,6), C7).mean():.1f}")
