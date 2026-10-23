"""
Statistics-source screen for the k3 sparse mask (STM input-conditioning side-quest, paper repo plan.md section 9).
No training. The k3 key is the step-1 observation [e128, bias = 0, logits] of the frozen LTM, so it does not depend on
the STM: this rebuilds the conditioned runs' actual sparse masks exactly, with the checkpoint's own projections.

Question: the runs centre the key with statistics fixed on the STM pre-training images (coarse 0,1, fine groups 1,2).
Do the new classes (groups 3,4,5) then keep a common offset that puts them on overlapping masks, and does a broad
statistics set fix that? Statistic sources (disjoint from the measured images):
    task12    coarse 0,1, groups 1,2       what the runs use (estimated during STM pre-training)
    task345   coarse 0,1, groups 3,4,5     oracle (future classes), reference only
    other18   coarse 2-19, groups 1-5      broad, no task images: the candidate
    all20     coarse 0-19, groups 1-5      broad incl. future task classes, reference
    ltmseen   coarse 0-19, groups 1,2      the LTM's own training classes
Measured on coarse 0,1 groups 1-5, training split (no test images). Also writes the other18 statistics in
InputConditioner format (for --input-stats-first-file) and checks task12 against the checkpoint's stored statistics.

Usage (repo root):
    python stm_mask_screen_stats.py --ltm-checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth \
        --base-stm-checkpoint runs_v2/pretrain/rl/pair0_1/seed1/stm_pretrain.pth \
        --cond-stm-checkpoint runs_cond/pretrain/centre_mask_first/pair0_1/seed1/stm_pretrain.pth \
        --out-dir runs_local/mask_screen_stats_20261010
"""
import argparse
import json
import logging
import os
import types

import numpy as np
import torch

import analyze_stm_inspectability as A
from util.sparse import get_ensemble_sparse_signed_pairs_projection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARTS = {"e128": slice(0, 128), "bias": slice(128, 256), "logits": slice(256, 276)}
PART_SIZES = [128, 128, 20]
OBS_SIZE = 276
H, K = 1000, 32
TASK_COARSE = [0, 1]
SOURCES = {
    "task12": (TASK_COARSE, [1, 2]),
    "task345": (TASK_COARSE, [3, 4, 5]),
    "other18": (list(range(2, 20)), [1, 2, 3, 4, 5]),
    "all20": (list(range(20)), [1, 2, 3, 4, 5]),
    "ltmseen": (list(range(20)), [1, 2]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ltm-checkpoint", required=True)
    p.add_argument("--base-stm-checkpoint", required=True, help="any original-input STM checkpoint (to build the LTM)")
    p.add_argument("--cond-stm-checkpoint", required=True, help="a conditioned k3 STM checkpoint (projections, stats)")
    p.add_argument("--data-path", default="../cifar-100-python")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fit-per-label", type=int, default=100)
    p.add_argument("--eval-per-label", type=int, default=100)
    p.add_argument("--projection-seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def step1_observations(model, images, batch_size):
    out = []
    for s in range(0, images.shape[0], batch_size):
        image = images[s:s + batch_size]
        bias = torch.zeros(image.shape[0], 128)
        logits, _, e128 = A.ltm_pass(model, image, bias)
        out.append(torch.cat([e128, bias, logits], dim=1))
    return torch.cat(out).numpy().astype(np.float64)


def fit(X):
    mu = X.mean(0)
    sq = np.array([(X[:, s] ** 2).sum(1).mean() for s in PARTS.values()])        # E ||x_p||^2 (uncentred)
    centred = np.array([((X[:, s] - mu[s]) ** 2).sum(1).mean() for s in PARTS.values()])
    scale = np.ones(OBS_SIZE)
    for (name, s), c in zip(PARTS.items(), centred):
        scale[s] = 1 / np.sqrt(c) if c > 1e-10 else 1.0
    return {"mu": mu, "part_sq_norm": sq, "scale": scale, "n": X.shape[0]}


def condition(X, how, st):
    if how == "raw":
        return X
    if how == "c1":
        return X - st["mu"]
    if how == "c2":
        return (X - st["mu"]) * st["scale"]
    raise ValueError(how)


def masks(key, P):
    return torch.topk(torch.from_numpy(key.astype(np.float32)) @ P.T, K, dim=1).indices.numpy()


def to_sets(idx):
    S = np.zeros((idx.shape[0], H), dtype=bool)
    np.put_along_axis(S, idx, True, axis=1)
    return S


def jac(a, b):
    return (a & b).sum(-1) / np.maximum((a | b).sum(-1), 1)


def measure(S, yc, yf, g, rng, pairs=40000):
    new = np.isin(g, [3, 4, 5])
    m = {"units_all": int(S.any(0).sum()), "units_new345": int(S[new].any(0).sum())}
    for gg in (1, 2, 3, 4, 5):
        m[f"units_g{gg}"] = int(S[g == gg].any(0).sum())
    # pairs among the new classes
    idx = np.flatnonzero(new)
    i, j = rng.choice(idx, pairs), rng.choice(idx, pairs)
    ok = i != j; i, j = i[ok], j[ok]
    J = jac(S[i], S[j]); sf = yf[i] == yf[j]; sc = yc[i] == yc[j]
    m["J345_same_fine"] = float(J[sf].mean())
    m["J345_same_coarse_diff_fine"] = float(J[sc & ~sf].mean())
    m["J345_diff_coarse"] = float(J[~sc].mean())
    # usage overlap between groups (histogram intersection of cell-usage distributions, 1 = identical)
    use = {gg: S[g == gg].sum(0) / S[g == gg].sum() for gg in (1, 2, 3, 4, 5)}
    u12 = S[np.isin(g, [1, 2])].sum(0); u12 = u12 / u12.sum()
    for a, b in ((3, 4), (3, 5), (4, 5)):
        m[f"usage_overlap_g{a}g{b}"] = float(np.minimum(use[a], use[b]).sum())
    for gg in (3, 4, 5):
        m[f"usage_overlap_g{gg}_vs_g12"] = float(np.minimum(use[gg], u12).sum())
    counts = S.sum(0); m["top100_share"] = float(np.sort(counts)[::-1][:100].sum() / counts.sum())
    return m


def offset(Xc, g):
    """Common offset of the new classes after conditioning: |mean of groups 3-5| / RMS norm of their keys."""
    Z = Xc[np.isin(g, [3, 4, 5])]
    return float(np.linalg.norm(Z.mean(0)) / np.sqrt((Z ** 2).sum(1).mean()))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    la = types.SimpleNamespace(data_path=args.data_path, coarse_classes=list(range(20)), fine_groups=[1, 2, 3, 4, 5],
                               split="train", max_per_label=None)
    images, y_coarse, y_fine, groups = A.load_images(la, rng)

    # Per fine label: the first fit_per_label images go to statistics, the next eval_per_label (task only) to measuring
    fit_pool, eval_idx = [], []
    for lab in np.unique(y_fine):
        idx = rng.permutation(np.flatnonzero(y_fine == lab))
        fit_pool += list(idx[:args.fit_per_label])
        if y_coarse[idx[0]] in TASK_COARSE:
            eval_idx += list(idx[args.fit_per_label:args.fit_per_label + args.eval_per_label])
    fit_pool, eval_idx = np.array(fit_pool), np.array(eval_idx)
    use = np.union1d(fit_pool, eval_idx)
    logger.info(f"step-1 observations for {len(use)} images ({len(eval_idx)} measured)")

    model = A.build_model(types.SimpleNamespace(ltm_checkpoint=args.ltm_checkpoint,
                                                stm_checkpoint=args.base_stm_checkpoint,
                                                sparsity=K, actor_training="rl"), torch.device("cpu"))
    X_all = np.zeros((len(images), OBS_SIZE))
    X_all[use] = step1_observations(model, images[use], args.batch_size)
    assert np.all(X_all[use][:, PARTS["bias"]] == 0)

    stats = {}
    for name, (coarse, grp) in SOURCES.items():
        sel = fit_pool[np.isin(y_coarse[fit_pool], coarse) & np.isin(groups[fit_pool], grp)]
        stats[name] = fit(X_all[sel])
        logger.info(f"stats {name}: {len(sel)} images")

    # Check: task12 against the statistics the conditioned run estimated during its STM pre-training
    ck = torch.load(args.cond_stm_checkpoint, map_location="cpu", weights_only=True)
    ck_first = ck["input_conditioner_first"]
    mu_ck = ck_first["mean"].double().numpy()
    check = {"ckpt_samples": int(ck_first["samples"]),
             "ckpt_part_sq_norm": ck_first["part_sq_norm"].tolist(),
             "task12_part_sq_norm": stats["task12"]["part_sq_norm"].tolist()}
    for name, st in stats.items():
        d = st["mu"] - mu_ck
        check[f"mean_distance_to_ckpt_{name}"] = {
            p: float(np.linalg.norm(d[s]) / max(np.linalg.norm(mu_ck[s]), 1e-12)) for p, s in PARTS.items() if p != "bias"}
    logger.info(f"checkpoint check: {json.dumps(check)}")

    # Statistics file for training (InputConditioner state, 'first' conditioner)
    st = stats["other18"]
    torch.save({"source": "other18: coarse 2-19, fine groups 1-5, training split, step-1 (bias 0) observations",
                "mean": torch.tensor(st["mu"], dtype=torch.float32),
                "part_sq_norm": torch.tensor(st["part_sq_norm"], dtype=torch.float32),
                "samples": int(st["n"])}, os.path.join(args.out_dir, "first_stats_other18.pt"))

    projections = {"actor_ckpt": ck["model_actor"]["input_projection"].float(),
                   "critic_ckpt": ck["model_critic"]["input_projection"].float()}
    for s in args.projection_seeds:
        torch.manual_seed(s)
        projections[f"fresh{s}"] = get_ensemble_sparse_signed_pairs_projection(1, OBS_SIZE, H, device="cpu")[0].float()

    Xe = X_all[eval_idx]; yc, yf, g = y_coarse[eval_idx], y_fine[eval_idx], groups[eval_idx]
    results = []
    for how in ("raw", "c1", "c2"):
        for src in (["-"] if how == "raw" else list(SOURCES)):
            Xc = condition(Xe, how, stats.get(src))
            for pname, P in projections.items():
                m = measure(to_sets(masks(Xc, P)), yc, yf, g, np.random.default_rng(0))
                results.append({"cond": how, "stats": src, "projection": pname, "offset345": offset(Xc, g), **m})
            r = results[-len(projections)]
            logger.info(f"{how} {src}: actor units all {r['units_all']} new {r['units_new345']} "
                        f"J345 coarse-diff-fine {r['J345_same_coarse_diff_fine']:.3f} diff-coarse {r['J345_diff_coarse']:.3f} "
                        f"overlap g3g4 {r['usage_overlap_g3g4']:.3f} offset {r['offset345']:.3f}")

    json.dump({"check": check, "results": results}, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    write_report(results, check, os.path.join(args.out_dir, "report.md"))


def write_report(results, check, path):
    cols = ["cond", "stats", "offset345", "units_all", "units_g1", "units_g2", "units_g3", "units_g4", "units_g5",
            "units_new345", "J345_same_fine", "J345_same_coarse_diff_fine", "J345_diff_coarse",
            "usage_overlap_g3g4", "usage_overlap_g3g5", "usage_overlap_g4g5",
            "usage_overlap_g3_vs_g12", "usage_overlap_g4_vs_g12", "usage_overlap_g5_vs_g12", "top100_share"]
    lines = [f"# k3 mask: statistics-source screen (random k-of-H Jaccard {K / (2 * H - K):.4f})", "",
             "Checkpoint check: " + json.dumps(check), ""]
    for pname in dict.fromkeys(r["projection"] for r in results):
        lines += [f"## projection: {pname}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in results:
            if r["projection"] == pname:
                lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        lines.append("")
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
