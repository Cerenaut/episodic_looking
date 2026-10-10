"""
Key-parts screen for the k3 sparse mask (STM input-conditioning side-quest, paper repo plan.md section 9). No training.
Question (Dave, 10 Oct): do the logits dominate the mask key and pull fine classes 3,4,5 onto the same cells? Rebuilds
the conditioned run's k3 masks (its own actor projection + a fresh one) with the logit part of the key weighted
1 / 0.5 / 0.25 / 0 (0 = zeroed, key stays 276 wide), and compares:
  - class-level isolation: overlap of the cell-usage distributions of fine 3 vs 4 vs 5, against a null with the
    fine labels shuffled within each coarse class (no class structure at population level)
  - pairwise mask overlap (Jaccard) for same-coarse different-fine image pairs
  - information kept: 5-fold linear-probe accuracy for fine 3/4/5 within each coarse class (chance 1/3) from the
    mask (1000 on/off cells) vs from the inputs it is computed from (e128, logits, both). If the mask keeps much less
    than its input has, the key/projection loses class detail (fixable); if about the same, the limit is the LTM.
Training-split images; statistics fitted on coarse 0,1 fine groups 1,2 (as in the runs), disjoint from measured images.

Usage (repo root):
    python stm_mask_screen_parts.py --ltm-checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth \
        --base-stm-checkpoint runs_v2/pretrain/rl/pair0_1/seed1/stm_pretrain.pth \
        --cond-stm-checkpoint runs_cond/pretrain/centre_mask_first/pair0_1/seed1/stm_pretrain.pth \
        --out-dir runs_local/mask_screen_parts_20261010
"""
import argparse
import json
import logging
import os
import types

import numpy as np
import torch

import analyze_stm_inspectability as A
from stm_mask_screen_stats import PARTS, OBS_SIZE, H, K, fit, condition, masks, to_sets, jac, step1_observations
from util.sparse import get_ensemble_sparse_signed_pairs_projection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
NEW = [3, 4, 5]
LOGIT_WEIGHTS = [1.0, 0.5, 0.25, 0.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ltm-checkpoint", required=True)
    p.add_argument("--base-stm-checkpoint", required=True)
    p.add_argument("--cond-stm-checkpoint", required=True)
    p.add_argument("--data-path", default="../cifar-100-python")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fit-per-label", type=int, default=100)
    p.add_argument("--eval-per-label", type=int, default=300)
    p.add_argument("--null-shuffles", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def usage_overlap(S, labels, a, b):
    ua = S[labels == a].sum(0); ub = S[labels == b].sum(0)
    return float(np.minimum(ua / ua.sum(), ub / ub.sum()).sum())


def class_isolation(S, yc, yf, rng, shuffles):
    """Mean usage overlap over same-coarse pairs of new fine classes, and the label-shuffled null."""
    def mean_overlap(labels):
        vals = []
        for c in np.unique(yc):
            labs = np.unique(labels[yc == c])
            vals += [usage_overlap(S, labels, a, b) for i, a in enumerate(labs) for b in labs[i + 1:]]
        return float(np.mean(vals))
    real = mean_overlap(yf)
    null = []
    for _ in range(shuffles):
        yf_s = yf.copy()
        for c in np.unique(yc):
            idx = np.flatnonzero(yc == c); yf_s[idx] = rng.permutation(yf[idx])
        null.append(mean_overlap(yf_s))
    return real, float(np.mean(null))


def pair_jaccard(S, yc, yf, rng, pairs=40000):
    i, j = rng.integers(0, len(S), pairs), rng.integers(0, len(S), pairs)
    keep = (i != j) & (yc[i] == yc[j]) & (yf[i] != yf[j])
    return float(jac(S[i[keep]], S[j[keep]]).mean())


def probe(X, yc, yf, seed):
    """Fine 3/4/5 within each coarse class (chance 1/3), averaged over the coarse classes."""
    return float(np.mean([A.linear_probe(X[yc == c], yf[yc == c], seed) for c in np.unique(yc)]))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(args.seed)

    la = types.SimpleNamespace(data_path=args.data_path, coarse_classes=[0, 1], fine_groups=[1, 2, 3, 4, 5],
                               split="train", max_per_label=None)
    images, y_coarse, y_fine, groups = A.load_images(la, rng)
    fit_idx, eval_idx = [], []
    for lab in np.unique(y_fine):
        idx = rng.permutation(np.flatnonzero(y_fine == lab))
        if groups[idx[0]] in (1, 2):
            fit_idx += list(idx[:args.fit_per_label])
        else:
            eval_idx += list(idx[:args.eval_per_label])
    fit_idx, eval_idx = np.array(fit_idx), np.array(eval_idx)
    logger.info(f"statistics from {len(fit_idx)} images (groups 1,2); measured {len(eval_idx)} images (groups 3-5)")

    model = A.build_model(types.SimpleNamespace(ltm_checkpoint=args.ltm_checkpoint, stm_checkpoint=args.base_stm_checkpoint,
                                                sparsity=K, actor_training="rl"), torch.device("cpu"))
    st = fit(step1_observations(model, images[fit_idx], 250))
    X = step1_observations(model, images[eval_idx], 250)
    yc, yf = y_coarse[eval_idx], y_fine[eval_idx]

    ck = torch.load(args.cond_stm_checkpoint, map_location="cpu", weights_only=True)
    torch.manual_seed(1)
    projections = {"actor_ckpt": ck["model_actor"]["input_projection"].float(),
                   "fresh1": get_ensemble_sparse_signed_pairs_projection(1, OBS_SIZE, H, device="cpu")[0].float()}

    inputs = {"e128": X[:, PARTS["e128"]], "logits": X[:, PARTS["logits"]],
              "e128+logits": np.concatenate([X[:, PARTS["e128"]], X[:, PARTS["logits"]]], 1)}
    input_probe = {k: probe(v, yc, yf, args.seed) for k, v in inputs.items()}
    logger.info(f"input probes (chance 0.333): {json.dumps({k: round(v, 3) for k, v in input_probe.items()})}")

    results = []
    for how in ("c1", "c2"):
        Xc = condition(X, how, st)
        for lw in LOGIT_WEIGHTS:
            key = Xc.copy(); key[:, PARTS["logits"]] *= lw
            share = float((key[:, PARTS["logits"]] ** 2).sum(1).mean() / (key ** 2).sum(1).mean())
            for pname, P in projections.items():
                S = to_sets(masks(key, P))
                iso, null = class_isolation(S, yc, yf, np.random.default_rng(0), args.null_shuffles)
                r = {"cond": how, "logit_weight": lw, "projection": pname, "logit_share_of_key": share,
                     "units_used": int(S.any(0).sum()), "usage_overlap_new": iso, "usage_overlap_null": null,
                     "J_same_coarse_diff_fine": pair_jaccard(S, yc, yf, np.random.default_rng(0)),
                     "mask_probe": probe(S.astype(np.float32), yc, yf, args.seed)}
                results.append(r)
                logger.info(f"{how} logits x{lw} {pname}: logit share {share:.2f}, units {r['units_used']}, "
                            f"usage overlap {iso:.3f} (null {null:.3f}), J {r['J_same_coarse_diff_fine']:.3f}, "
                            f"mask probe {r['mask_probe']:.3f}")

    json.dump({"input_probe": input_probe, "results": results}, open(os.path.join(args.out_dir, "results.json"), "w"), indent=1)
    cols = ["cond", "logit_weight", "projection", "logit_share_of_key", "units_used", "usage_overlap_new",
            "usage_overlap_null", "J_same_coarse_diff_fine", "mask_probe"]
    lines = ["# k3 mask: logits in the key (fine 3,4,5 within coarse 0,1; chance probe 0.333)", "",
             "Input probes: " + ", ".join(f"{k} {v:.3f}" for k, v in input_probe.items()), "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |" for r in results]
    open(os.path.join(args.out_dir, "report.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
