"""
C.14 + B.6 hybrid, cheapest form (research_plan.md phase 1 item 3, motivated by the phase-2 oracle-template result):
a prototype (NCM) read-out on the LTM's unbiased features picks the fine label, and that label's class recipe (the STM's
mean final-step bias for the label, a 128-channel gain) is applied as a fixed gate; the LTM's biased logits give the
20-way prediction. No STM in the loop at test time, no training: templates and prototypes come from a train-split dump
of analyze_stm_inspectability.py, evaluation uses its test-split dump and the LTM model.

    python hybrid_template_ncm.py --train-dump runs_local/inspect_rl_fixed_c345_train/dump.npz \
        --test-dump runs_local/inspect_rl_fixed_c345/dump.npz --ltm-checkpoint ... --stm-checkpoint ... --actor-training rl

Reports, per fine-class group and overall: unbiased LTM, full STM episode (from the test dump), NCM alone (coarse
prediction of the nearest fine-label prototype), the hybrid (template of the NCM-chosen label), the oracle (template of
the true label; train templates, so a fair version of the report's oracle), and the hybrid with prototypes on the
128-d stage-2 input the STM reads instead of the 512-d features.
"""
import argparse
import logging

import numpy as np
import torch

from analyze_stm_inspectability import build_model, load_images, ltm_pass
from util.device import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dump", required=True)
    p.add_argument("--test-dump", required=True)
    p.add_argument("--ltm-checkpoint", required=True)
    p.add_argument("--stm-checkpoint", required=True)
    p.add_argument("--actor-training", default="rl", choices=["rl", "differentiable"])
    p.add_argument("--sparsity", type=int, default=32)
    p.add_argument("--data-path", default="../cifar-100-python")
    p.add_argument("--coarse-classes", type=int, nargs="+", default=[0, 1])
    p.add_argument("--fine-groups", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--split", default="test")
    p.add_argument("--max-per-label", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def prototypes(X, y):
    labs = np.unique(y)
    P = np.stack([X[y == l].mean(0) for l in labs])
    return P / np.linalg.norm(P, axis=1, keepdims=True), labs


def nearest(X, P, labs):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return labs[(Xn @ P.T).argmax(1)]


def gate_predict(model, images, bias, batch_size, device):
    preds = []
    with torch.no_grad():
        for s in range(0, len(images), batch_size):
            lg, _, _ = ltm_pass(model, images[s:s + batch_size].to(device),
                                torch.as_tensor(bias[s:s + batch_size], dtype=torch.float32, device=device))
            preds.append(lg.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def by_group(pred, y, groups):
    return {g: float((pred[groups == g] == y[groups == g]).mean()) for g in np.unique(groups)} | {"all": float((pred == y).mean())}


def main():
    args = parse_args()
    device = get_device()
    tr, te = dict(np.load(args.train_dump)), dict(np.load(args.test_dump))
    fine_labs = np.unique(tr["y_fine"])
    fine_to_coarse = {int(l): int(tr["y_coarse"][tr["y_fine"] == l][0]) for l in fine_labs}

    # class recipes and prototypes from the train split
    templates = np.stack([tr["bias"][tr["y_fine"] == l][:, -1].mean(0) for l in fine_labs])  # [L, 128]
    P512, _ = prototypes(tr["feat0"], tr["y_fine"])
    P128, _ = prototypes(tr["e0"], tr["y_fine"])

    # test images in the same order as the test dump (same loader, same seed as the dump run)
    rng = np.random.default_rng(args.seed)
    images, y_coarse, y_fine, groups = load_images(args, rng)
    assert (y_fine == te["y_fine"]).all(), "test dump and reloaded images differ; use the dump's --seed/--max-per-label"
    model = build_model(args, device)

    rows = {}
    rows["unbiased LTM"] = by_group(te["logits0"].argmax(-1), y_coarse, groups)
    rows["full STM episode"] = by_group(te["logits"][:, -1].argmax(-1), y_coarse, groups)
    for name, X_te, P in [("512-d features", te["feat0"], P512), ("128-d stage-2 input", te["e0"], P128)]:
        pick = nearest(X_te, P, fine_labs)
        ncm_coarse = np.vectorize(fine_to_coarse.get)(pick)
        rows[f"NCM alone ({name}): coarse of nearest fine-label prototype"] = by_group(ncm_coarse, y_coarse, groups)
        rows[f"  fine-label (10-way) accuracy of that NCM ({name})"] = by_group(pick, y_fine, groups)
        bias = templates[np.searchsorted(fine_labs, pick)]
        rows[f"hybrid: template of NCM-chosen label ({name}), LTM predicts"] = by_group(gate_predict(model, images, bias, args.batch_size, device), y_coarse, groups)
    own = templates[np.searchsorted(fine_labs, y_fine)]
    rows["oracle: template of the true label (train templates)"] = by_group(gate_predict(model, images, own, args.batch_size, device), y_coarse, groups)
    glob = np.repeat(templates.mean(0, keepdims=True), len(images), 0)
    rows["one global template (train)"] = by_group(gate_predict(model, images, glob, args.batch_size, device), y_coarse, groups)

    gl = sorted(g for g in rows["unbiased LTM"] if g != "all")
    print("| gate | " + " | ".join(f"group {g}" for g in gl) + " | all |")
    print("|---|" + "---|" * (len(gl) + 1))
    for k, v in rows.items():
        print(f"| {k} | " + " | ".join(f"{v[g]:.3f}" for g in gl) + f" | {v['all']:.3f} |")


if __name__ == "__main__":
    main()
