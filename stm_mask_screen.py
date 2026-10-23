"""
Mask (address) screen for the STM input-conditioning side-quest (paper repo Notes/experiments/plan.md section 9).
No training. Replays the 8-step episode of an existing RL STM checkpoint on training-split images (no test images),
records each step's observation [e128, bias, logits] (276 values), then rebuilds the actor's sparse mask under
alternative addresses and measures it:
    conditioning   raw | c1 (centre per feature) | c2 (c1 + one RMS scale per part) | c3 (per-sample LayerNorm per part)
    key source     k1 (8-step history, as now) | k2 (current step only) | k3 (step-1 observation, the unbiased pass)
    bias weight    1 | 0 (bias part of the key zeroed after conditioning)
Statistics for c1/c2 are fitted per key source (k3: step-1 observations only) on fine groups 1,2 (the STM pre-training classes) under the sampled bias, as they
would be if estimated during STM pre-training. Masks are measured on a disjoint set of all five groups.
The projection is a fresh fixed random projection of the same kind as the model's (util/sparse.py), several seeds;
the checkpoint's own projection is reported for raw k1 as the reference (and checked to reproduce the actor's masks).

Interpretation limits (plan.md 9.9): inputs are recorded under a baseline STM trained with the old address, so this
measures the address on the wrong input distribution and says nothing about accuracy. k3's within-episode stability
and train/eval agreement hold by construction.

Usage (repo root):
    python stm_mask_screen.py --ltm-checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth \
        --stm-checkpoints runs_v2/pretrain/rl/pair0_1/seed1/stm_pretrain.pth \
                          runs_v2/continual/rl/pair0_1/seed1/order3_4_5/stm_phase5.pth \
        --out-dir runs_local/mask_screen_20261009
"""
import argparse
import json
import logging
import os
import types

import numpy as np
import torch

import analyze_stm_inspectability as A
from util.observation_history import ObservationHistory, ObservationHistoryConfig
from util.sparse import get_ensemble_sparse_signed_pairs_projection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BIAS_SIZE = 128
PARTS = {"e128": slice(0, 128), "bias": slice(128, 256), "logits": slice(256, 276)}
OBS_SIZE = 276
T = A.CONTEXT_SIZE  # 8
H = 1000
K = 32


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ltm-checkpoint", required=True)
    p.add_argument("--stm-checkpoints", nargs="+", required=True)
    p.add_argument("--data-path", default="../cifar-100-python")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fit-per-label", type=int, default=100)
    p.add_argument("--eval-per-label", type=int, default=60)
    p.add_argument("--projection-seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--policy-std", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------------------------------------
# Episode replay: records the observation entering the history at each step, and the actor's masks
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def replay(model, images, mode: str, std: float, batch_size: int, gen: torch.Generator):
    """mode 'mean': evaluation episode (bias = policy mean). 'sample': training-like (mean + N(0, std)).
    Returns obs [N, T, 276] (obs[:, 0] is the unbiased pass) and the actor's recorded mask indices [N, T, K]."""
    probe = A.StmProbe(model)
    obs_all, idx_all = [], []
    for start in range(0, images.shape[0], batch_size):
        image = images[start:start + batch_size]
        B = image.shape[0]
        history = ObservationHistory(
            ObservationHistoryConfig(batch_size=B, history_size=T, observation_size=OBS_SIZE), device=model.device)
        bias = torch.zeros(B, BIAS_SIZE)
        logits, _, e128 = A.ltm_pass(model, image, bias)
        obs_b, idx_b = [], []
        for t in range(T):
            o = torch.cat([e128, bias, logits], dim=1)
            obs_b.append(o)
            history.update(o)
            mean, idx, _ = probe.actor(history.get_tensor_vector())
            idx_b.append(idx)
            if mode == "sample":
                bias = mean + std * torch.randn(mean.shape, generator=gen)
            else:
                bias = mean.clone()
            logits, _, e128 = A.ltm_pass(model, image, bias)
        obs_all.append(torch.stack(obs_b, 1)); idx_all.append(torch.stack(idx_b, 1))
    return torch.cat(obs_all).numpy(), torch.cat(idx_all).numpy()


# ------------------------------------------------------------------------------------------------
# Conditioning (fitted on the fit set, sampled bias, all steps pooled)
# ------------------------------------------------------------------------------------------------
def fit_stats(obs_fit):
    X = obs_fit.reshape(-1, OBS_SIZE).astype(np.float64)
    mu = X.mean(0)
    scale = np.ones(OBS_SIZE)
    for name, s in PARTS.items():
        rms_norm = np.sqrt(((X[:, s] - mu[s]) ** 2).sum(1).mean())  # expected norm of the centred part
        scale[s] = 1.0 / max(rms_norm, 1e-8)
    return {"mu": mu.astype(np.float32), "scale": scale.astype(np.float32),
            "part_norm_raw": {n: float(np.sqrt((X[:, s] ** 2).sum(1).mean())) for n, s in PARTS.items()},
            "part_norm_centred": {n: float(1 / scale[s][0]) for n, s in PARTS.items()}}


def condition(obs, how: str, stats, bias_weight: float):
    """obs [..., 276] -> conditioned copy."""
    x = obs.astype(np.float32).copy()
    if how == "raw":
        pass
    elif how == "c1":
        x = x - stats["mu"]
    elif how == "c2":
        x = (x - stats["mu"]) * stats["scale"]
    elif how == "c3":
        for s in PARTS.values():
            p = x[..., s]
            m = p.mean(-1, keepdims=True); sd = p.std(-1, keepdims=True)
            x[..., s] = (p - m) / (sd + 1e-5)
    else:
        raise ValueError(how)
    x[..., PARTS["bias"]] *= bias_weight
    return x


def keys(xc, source: str):
    """xc [N, T, 276] conditioned observations -> keys [N, T, D] for each step t (the key the actor reads at step t)."""
    N = xc.shape[0]
    if source == "k2":
        return xc
    if source == "k3":
        return np.repeat(xc[:, :1], T, axis=1)
    # k1: history; at step t slots T-1-t .. T-1 hold obs 0..t, earlier slots empty (0, i.e. the mean after centring)
    out = np.zeros((N, T, T * OBS_SIZE), dtype=np.float32)
    for t in range(T):
        hist = np.zeros((N, T, OBS_SIZE), dtype=np.float32)
        hist[:, T - 1 - t:] = xc[:, :t + 1]
        out[:, t] = hist.reshape(N, -1)
    return out


def topk_masks(key, projection):
    """key [N, T, D], projection [H, D] -> indices [N, T, K]"""
    N, Tt, D = key.shape
    scores = torch.from_numpy(key.reshape(-1, D)) @ projection.T
    return torch.topk(scores, K, dim=1).indices.reshape(N, Tt, K).numpy()


def make_projection(D, seed):
    torch.manual_seed(seed)
    return get_ensemble_sparse_signed_pairs_projection(1, D, H, device="cpu")[0]


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------
def to_sets(idx):
    sets = np.zeros(idx.shape[:-1] + (H,), dtype=bool)
    np.put_along_axis(sets, idx, True, axis=-1)
    return sets


def jac(a, b):
    return (a & b).sum(-1) / np.maximum((a | b).sum(-1), 1)


def metrics(idx_mean, idx_sample, y_coarse, y_fine, rng, pairs=20000):
    S = to_sets(idx_mean)          # [N, T, H]
    Ss = to_sets(idx_sample)
    N = S.shape[0]
    last = S[:, -1]
    i = rng.integers(0, N, pairs); j = rng.integers(0, N, pairs); ok = i != j; i, j = i[ok], j[ok]
    J = jac(last[i], last[j])
    same_fine = y_fine[i] == y_fine[j]; same_coarse = y_coarse[i] == y_coarse[j]
    counts = S.reshape(-1, H).sum(0)
    top100 = np.sort(counts)[::-1][:100].sum() / counts.sum()
    m = {
        "units_used_last_step": int(last.any(0).sum()),
        "units_used_all_steps": int(S.any((0, 1)).sum()),
        "top100_share": float(top100),
        "J_same_fine": float(J[same_fine].mean()),
        "J_same_coarse_diff_fine": float(J[same_coarse & ~same_fine].mean()),
        "J_diff_coarse": float(J[~same_coarse].mean()),
        "J_step1_vs_step8": float(jac(S[:, 0], S[:, -1]).mean()),
        "J_consecutive": float(np.mean([jac(S[:, t], S[:, t + 1]).mean() for t in range(T - 1)])),
        "J_train_vs_eval": float(np.mean([jac(S[:, t], Ss[:, t]).mean() for t in range(T)])),
    }
    m["sep_ratio_fine_vs_diffcoarse"] = m["J_same_fine"] / max(m["J_diff_coarse"], 1e-9)
    return m


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cpu")

    # Images: training split only (no test images). Fit: groups 1,2; eval: all groups, disjoint images.
    la = types.SimpleNamespace(data_path=args.data_path, coarse_classes=[0, 1], fine_groups=[1, 2, 3, 4, 5],
                               split="train", max_per_label=None)
    images, y_coarse, y_fine, groups = A.load_images(la, rng)
    fit_idx, eval_idx = [], []
    for lab in np.unique(y_fine):
        idx = rng.permutation(np.flatnonzero(y_fine == lab))
        g = groups[idx[0]]
        n_fit = args.fit_per_label if g in (1, 2) else 0
        fit_idx += list(idx[:n_fit]); eval_idx += list(idx[n_fit:n_fit + args.eval_per_label])
    fit_idx, eval_idx = np.array(fit_idx), np.array(eval_idx)
    logger.info(f"fit {len(fit_idx)} images (groups 1,2), eval {len(eval_idx)} images (groups 1-5)")

    results = []
    for ckpt in args.stm_checkpoints:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        logger.info(f"checkpoint {ckpt}")
        model = A.build_model(types.SimpleNamespace(ltm_checkpoint=args.ltm_checkpoint, stm_checkpoint=ckpt,
                                                    sparsity=K, actor_training="rl"), device)
        obs_fit, _ = replay(model, images[fit_idx], "sample", args.policy_std, args.batch_size, gen)
        obs_ev, idx_ev_rec = replay(model, images[eval_idx], "mean", args.policy_std, args.batch_size, gen)
        obs_ev_s, idx_ev_s_rec = replay(model, images[eval_idx], "sample", args.policy_std, args.batch_size, gen)
        np.savez_compressed(os.path.join(args.out_dir, f"obs_{name}.npz"), obs_fit=obs_fit, obs_eval_mean=obs_ev,
                            obs_eval_sample=obs_ev_s, eval_idx=eval_idx, fit_idx=fit_idx,
                            y_coarse=y_coarse[eval_idx], y_fine=y_fine[eval_idx])

        # Sanity: raw k1 with the checkpoint's projection reproduces the actor's recorded masks
        P_ckpt = model.model_actor.input_projection.cpu()
        rebuilt = topk_masks(keys(condition(obs_ev, "raw", None, 1.0), "k1"), P_ckpt)
        same = np.mean([set(a) == set(b) for a, b in zip(rebuilt.reshape(-1, K), idx_ev_rec.reshape(-1, K))])
        logger.info(f"  rebuilt raw k1 masks equal the actor's: {same:.4f}")

        # Each key fits its statistics on the observations it reads: k3 only ever sees step 1 (bias 0)
        stats_by_key = {"k1": fit_stats(obs_fit), "k2": fit_stats(obs_fit), "k3": fit_stats(obs_fit[:, :1])}
        stats = stats_by_key["k1"]
        logger.info(f"  part norms raw {stats['part_norm_raw']} centred {stats['part_norm_centred']}")
        logger.info(f"  step-1 part norms centred {stats_by_key['k3']['part_norm_centred']}")
        yc, yf = y_coarse[eval_idx], y_fine[eval_idx]

        ref = metrics(idx_ev_rec, idx_ev_s_rec, yc, yf, np.random.default_rng(0))
        results.append({"checkpoint": name, "cond": "raw", "key": "k1", "bias_w": 1.0, "projection": "checkpoint",
                        "mask_reproduced": float(same), **ref})

        for cond in ["raw", "c1", "c2", "c3"]:
            for bw in [1.0, 0.0]:
                for src in ["k1", "k2", "k3"]:
                    xm = condition(obs_ev, cond, stats_by_key[src], bw)
                    xs = condition(obs_ev_s, cond, stats_by_key[src], bw)
                    km, ks = keys(xm, src), keys(xs, src)
                    per_seed = []
                    for ps in args.projection_seeds:
                        P = make_projection(km.shape[-1], ps)
                        per_seed.append(metrics(topk_masks(km, P), topk_masks(ks, P), yc, yf, np.random.default_rng(0)))
                    row = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
                    row_sd = {k + "_sd": float(np.std([m[k] for m in per_seed])) for k in per_seed[0]}
                    results.append({"checkpoint": name, "cond": cond, "key": src, "bias_w": bw,
                                    "projection": f"fresh x{len(args.projection_seeds)}", **row, **row_sd})
                    logger.info(f"  {cond} {src} bw={bw}: used {row['units_used_last_step']:.0f} "
                                f"Jdiff {row['J_diff_coarse']:.3f} Jfine {row['J_same_fine']:.3f} "
                                f"1v8 {row['J_step1_vs_step8']:.3f} tr/ev {row['J_train_vs_eval']:.3f}")
        with open(os.path.join(args.out_dir, f"stats_{name}.json"), "w") as f:
            json.dump({src: {k: v for k, v in st.items() if k.startswith("part_")} for src, st in stats_by_key.items()},
                      f, indent=1)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_table(results, os.path.join(args.out_dir, "report.md"))


def write_table(results, path):
    cols = ["cond", "key", "bias_w", "projection", "units_used_last_step", "units_used_all_steps", "top100_share",
            "J_same_fine", "J_same_coarse_diff_fine", "J_diff_coarse", "sep_ratio_fine_vs_diffcoarse",
            "J_step1_vs_step8", "J_consecutive", "J_train_vs_eval"]
    lines = [f"# Mask screen (random k-of-H Jaccard: {K / (2 * H - K):.4f})", ""]
    for ck in dict.fromkeys(r["checkpoint"] for r in results):
        lines += [f"## {ck}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in results:
            if r["checkpoint"] != ck:
                continue
            lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) and c not in ("bias_w",)
                                           else str(r[c]) for c in cols) + " |")
        lines.append("")
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
