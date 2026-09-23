"""
Inspectability analysis of the CLS/STM model (research_plan.md, phase 2).

Drives the STM + frozen LTM directly on tensors (no gym envs), replaying the 8-step evaluation episode
exactly as CifarAgent does with the deterministic mean bias (--eval-bias mean), but with the fine labels
in hand and every internal quantity recorded per step:

    bias      [N, T, 128]   actor mean output, applied to stage 2 of the LTM
    logits    [N, T, 20]    LTM classification with that bias
    feat      [N, T, 512]   pooled final-stage features (what the LTM classifier reads)
    e128      [N, T, 128]   pooled, gated stage-2 output (what the STM reads)
    mask      [N, T, k]     active hidden-unit indices of the actor (sparse model only)
    hidden    [N, T, L, H]  actor hidden-layer activations after masking (float16)
    value     [N, T]        critic value

plus the unbiased pass (step 0). From these it runs the analyses of research_plan.md phase 2:

    A1  mask overlap within / between classes           A2  hidden-unit class selectivity
    A3  which units changed between checkpoints         A4  lesion of one class's units (model in the loop)
    A5  linear decodability of fine class               B1  per-class bias templates
    B2  template sufficiency (model in the loop)        B3  steering: class A's template on class B
    B4  movement toward NCM prototypes in feature space B5  bias trajectory over the episode

Usage (from the repo root, episodic env):

    python analyze_stm_inspectability.py \
        --ltm-checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth \
        --stm-checkpoint ../cifar_100_pretrain/variants/stm_rl_fixed_e13_pt12.pth \
        --out-dir runs_local/inspect_rl_fixed_e13 --lesion --templates

    # dense ablation: --sparsity 0 with a dense STM checkpoint
    # A3: --compare-checkpoints pre.pth phase3.pth phase4.pth phase5.pth (written by
    #     cifar_main_stm_training.py --experiment-type continual --stm-checkpoint-out <prefix>)

Outputs: <out-dir>/dump.npz, report.md, *.png.
"""
import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_model import CifarModel, CifarModelConfig
from util.device import get_device
from util.observation_history import ObservationHistory, ObservationHistoryConfig
from util.reinforcement_learning.policy_util import PolicyUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 20
BIAS_STAGE = 2
CONTEXT_SIZE = 8
FINE_GROUPS = [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ltm-checkpoint", required=True)
    p.add_argument("--stm-checkpoint", required=True)
    p.add_argument("--data-path", default="../cifar-100-python")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--coarse-classes", type=int, nargs="+", default=[0, 1])
    p.add_argument("--fine-groups", type=int, nargs="+", default=FINE_GROUPS,
                   help="fine-class groups (1..5, as in the reference scripts) to include")
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument("--max-per-label", type=int, default=None, help="images per CIFAR fine label (test has 100)")
    p.add_argument("--sparsity", type=int, default=32, help="k of the STM (0 = dense ablation)")
    p.add_argument("--actor-training", choices=["rl", "differentiable"], default="rl")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse-dump", action="store_true", help="skip the model pass if dump.npz exists")
    # model-in-the-loop analyses
    p.add_argument("--lesion", action="store_true", help="A4: lesion each group's most selective units")
    p.add_argument("--lesion-n", type=int, default=64, help="units removed per lesioned group")
    p.add_argument("--templates", action="store_true", help="B2/B3: class-mean bias as a fixed gate")
    p.add_argument("--compare-checkpoints", nargs="+", default=None,
                   help="A3: STM checkpoints in training order (pre-training, then each continual phase)")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def fine_label_to_group() -> dict:
    return {lab: g for g in FINE_GROUPS for lab in Cifar100Dataset.get_fine_class_set(g)}


def release_shared_memory(dataset: Cifar100Dataset):
    """Cifar100Dataset keeps its arrays in multiprocessing shared memory and never frees them."""
    for shm in (dataset.shared_memory_images, dataset.shared_memory_labels_coarse, dataset.shared_memory_labels_fine):
        if shm is None:
            continue
        try:
            shm.close()
            shm.unlink()
        except Exception:  # noqa: BLE001
            pass


def load_images(args, rng: np.random.Generator):
    exclude_fine = Cifar100Dataset.get_fine_classes([g for g in FINE_GROUPS if g not in args.fine_groups])
    exclude_coarse = Cifar100Dataset.get_coarse_classes_excluded(args.coarse_classes)
    ds = Cifar100Dataset(
        file_path=args.data_path,
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=(args.split == "train"),
        exclude_classes_coarse=exclude_coarse,
        exclude_classes_fine=exclude_fine,
        max_instances=None,
        as_tensor=True,
    )
    images = ds.images.clone()
    y_coarse = np.asarray(ds.labels_coarse, dtype=np.int64)
    y_fine = np.asarray(ds.labels_fine, dtype=np.int64)
    release_shared_memory(ds)

    if args.max_per_label is not None:
        keep = []
        for lab in np.unique(y_fine):
            idx = np.flatnonzero(y_fine == lab)
            keep.append(rng.choice(idx, size=min(args.max_per_label, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        images, y_coarse, y_fine = images[keep], y_coarse[keep], y_fine[keep]

    groups = np.vectorize(fine_label_to_group().get)(y_fine)
    logger.info(f"{len(images)} images, coarse {np.unique(y_coarse).tolist()}, "
                f"fine labels {np.unique(y_fine).tolist()}, groups {np.unique(groups).tolist()}")
    return images, y_coarse, y_fine, groups


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(args, device) -> CifarModel:
    """Same configuration as cifar_main_stm_training.py (only the fields that affect inference matter)."""
    config = CifarModelConfig(
        encoder_ensemble_size=1,
        encoder_sparsity=args.sparsity,
        history_size=CONTEXT_SIZE,
        model_hidden_size=1000,
        model_hidden_size_factor=1,
        model_nonlinearity="leaky-relu",
        model_layers=3,
        discount_factor=0.5,
        normalize_advantage=True,
        normalize_advantage_epsilon=0.0001,
        normalize_advantage_clamp=None,
        reward_scale=1.0,
        reward_type=CifarModel.REWARD_TYPE_ACCURACY_IMPROVEMENT,
        loss_actor_scale=0.01,
        loss_critic_scale=1.0,
        loss_entropy_scale=1.0,
        loss_class_scale=0.01,
        loss_huber_delta=1.0,
        loss_actor_type=CifarModel.LOSS_TYPE_SLOW_CHANGE,
        loss_critic_type=CifarModel.LOSS_TYPE_HUBER,
        num_classes=NUM_CLASSES,
        classifier_model_file=args.ltm_checkpoint,
        classifier_bias_stage=BIAS_STAGE,
        policy_std=0.5,
        actor_training=args.actor_training,
        eval_bias="mean",
    )
    model = CifarModel(config=config, device=device)
    state_dict = torch.load(args.stm_checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.model_actor.eval()
    model.model_critic.eval()
    return model


def mlp_forward_capture(mlp, x: torch.Tensor, mask: torch.Tensor | None):
    """SparseActivationDenseModel / DenseModel forward, returning the masked hidden activations too."""
    if mlp.ln is not None:
        x = mlp.ln(x)
    hidden = []
    for layer in range(mlp.config.layers):
        x = mlp.layers[layer](x)
        is_hidden = layer < (mlp.config.layers - 1)
        f = mlp.f_hidden if is_hidden else mlp.f_output
        if f is not None:
            x = f(x)
        if is_hidden:
            if mask is not None:
                x = x * mask
            hidden.append(x)
    return x, hidden


class StmProbe:
    """One actor / critic call with the mask exposed and optionally lesioned."""

    def __init__(self, model: CifarModel, lesion_units: np.ndarray | None = None):
        self.model = model
        self.sparse = model.config.encoder_sparsity > 0
        self.lesion = None if lesion_units is None or len(lesion_units) == 0 else torch.as_tensor(lesion_units, device=model.device)

    def _mlp(self, which):
        m = self.model.model_actor if which == "actor" else self.model.model_critic
        return m, (m.get_model(0) if self.sparse else m)

    def _mask(self, container, mlp, x):
        if self.sparse:
            idx = container.encode(x)  # [B, k]
            mask = mlp.create_active_mask(x, active_indices=idx)  # [B, H]
        else:
            idx = None
            mask = torch.ones(x.shape[0], mlp.config.hidden_size, device=x.device, dtype=x.dtype)
        if self.lesion is not None:
            mask[:, self.lesion] = 0.0
        return idx, mask

    @torch.no_grad()
    def actor(self, x: torch.Tensor):
        container, mlp = self._mlp("actor")
        idx, mask = self._mask(container, mlp, x)
        logits, hidden = mlp_forward_capture(mlp, x, mask if (self.sparse or self.lesion is not None) else None)
        mean = PolicyUtil.get_policy_mean_continuous(
            logits=PolicyUtil.clamp_logits(logits, max_logit_magnitude=self.model.policy_config.max_logit_magnitude),
            std=self.model.policy_config.policy_std,
        )
        return mean, idx, hidden

    @torch.no_grad()
    def critic(self, x: torch.Tensor):
        container, mlp = self._mlp("critic")
        idx, mask = self._mask(container, mlp, x)
        out, _ = mlp_forward_capture(mlp, x, mask if self.sparse else None)
        return out.squeeze(1)


@torch.no_grad()
def ltm_pass(model: CifarModel, image: torch.Tensor, bias: torch.Tensor | None):
    feat, e128 = model.model_class.encoder(x=image, bias=bias)
    logits = model.model_class.classifier(feat)
    return logits, feat, e128


@torch.no_grad()
def run_episodes(model: CifarModel, probe: StmProbe, images: torch.Tensor, steps: int, batch_size: int,
                 device, record: bool = True) -> dict:
    """Replay of CifarAgent's evaluation episode with the mean bias. Step t=0 is the unbiased pass."""
    N = images.shape[0]
    bias_size = model.get_bias_size()
    obs_size = bias_size + bias_size + NUM_CLASSES
    out = {k: [] for k in ["bias", "logits", "feat", "e128", "mask", "hidden", "value", "logits0", "feat0", "e0"]}

    for start in range(0, N, batch_size):
        image = images[start:start + batch_size].to(device)
        B = image.shape[0]
        history = ObservationHistory(
            ObservationHistoryConfig(batch_size=B, history_size=CONTEXT_SIZE, observation_size=obs_size),
            device=device,
        )
        bias = torch.zeros(B, bias_size, device=device)  # gate = 2*sigmoid(0) = 1: no change
        rec = {k: [] for k in ["bias", "logits", "feat", "e128", "mask", "hidden", "value"]}

        logits, feat, e128 = ltm_pass(model, image, bias)
        out["logits0"].append(logits.cpu()); out["feat0"].append(feat.cpu()); out["e0"].append(e128.cpu())

        for t in range(steps):
            # obs_1: LTM state under the bias applied so far
            history.update(torch.cat([e128, bias, logits], dim=1))
            x = history.get_tensor_vector()
            mean, idx, hidden = probe.actor(x)
            bias = mean.detach().clone()
            # obs_2: LTM state under the new bias (this is what reward / accuracy are computed on)
            logits, feat, e128 = ltm_pass(model, image, bias)
            if record:
                rec["bias"].append(bias.cpu()); rec["logits"].append(logits.cpu())
                rec["feat"].append(feat.cpu()); rec["e128"].append(e128.cpu())
                rec["mask"].append(idx.cpu() if idx is not None else torch.zeros(B, 0, dtype=torch.long))
                rec["hidden"].append(torch.stack(hidden, dim=1).half().cpu())
                rec["value"].append(probe.critic(x).cpu())
            else:
                rec["logits"].append(logits.cpu())
        for k, v in rec.items():
            if v:
                out[k].append(torch.stack(v, dim=1))  # [B, T, ...]

    return {k: torch.cat(v, dim=0).numpy() for k, v in out.items() if v}


def accuracy_by(labels_pred: np.ndarray, y: np.ndarray, by: np.ndarray) -> dict:
    return {int(g): float((labels_pred[by == g] == y[by == g]).mean()) for g in np.unique(by)}


# --------------------------------------------------------------------------------------
# Analyses on the dump (no model needed)
# --------------------------------------------------------------------------------------
def jaccard_stats(mask_idx: np.ndarray, groups: np.ndarray, y_fine: np.ndarray, H: int, rng, pairs: int = 20000):
    """Mean Jaccard overlap of active sets for random pairs: same fine label / same group / different group."""
    N, k = mask_idx.shape
    sets = np.zeros((N, H), dtype=bool)
    np.put_along_axis(sets, mask_idx, True, axis=1)
    i = rng.integers(0, N, pairs); j = rng.integers(0, N, pairs)
    ok = i != j; i, j = i[ok], j[ok]
    inter = (sets[i] & sets[j]).sum(1); union = (sets[i] | sets[j]).sum(1)
    jac = inter / np.maximum(union, 1)
    same_fine = y_fine[i] == y_fine[j]
    same_group = groups[i] == groups[j]
    res = {
        "same fine label": float(jac[same_fine].mean()),
        "same group, different fine label": float(jac[same_group & ~same_fine].mean()),
        "different group": float(jac[~same_group].mean()),
        "random k-of-H": float(k / (2 * H - k)),
    }
    return res, sets


def unit_usage(sets: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[n_labels, H] frequency with which each unit is active for each label."""
    labs = np.unique(labels)
    return np.stack([sets[labels == l].mean(0) for l in labs]), labs


def selectivity(usage: np.ndarray) -> dict:
    """Per unit: normalised entropy of p(label | unit active). 0 = one label only, 1 = uniform."""
    p = usage / np.maximum(usage.sum(0, keepdims=True), 1e-12)
    used = usage.sum(0) > 0
    ent = -(p * np.log(np.maximum(p, 1e-12))).sum(0) / np.log(usage.shape[0])
    ent = ent[used]
    return {
        "units used": int(used.sum()),
        "units never active": int((~used).sum()),
        "median normalised entropy": float(np.median(ent)),
        "fraction entropy < 0.5": float((ent < 0.5).mean()),
        "fraction entropy < 0.25": float((ent < 0.25).mean()),
        "entropies": ent,
    }


def linear_probe(X: np.ndarray, y: np.ndarray, seed: int, folds: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    accs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def cosine_matrix(M: np.ndarray) -> np.ndarray:
    n = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    return n @ n.T


def variance_between_fraction(X: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of total variance of X explained by the label means."""
    mu = X.mean(0)
    total = ((X - mu) ** 2).sum()
    between = sum(((X[labels == l].mean(0) - mu) ** 2).sum() * (labels == l).sum() for l in np.unique(labels))
    return float(between / max(total, 1e-12))


def ncm_prototypes(feat: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labs = np.unique(y)
    P = np.stack([feat[y == l].mean(0) for l in labs])
    return P, labs


def cosine_to_prototypes(feat: np.ndarray, P: np.ndarray) -> np.ndarray:
    f = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    p = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return f @ p.T


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def plot_heatmap(M, row_labels, col_labels, title, path, cmap="viridis", vmin=None, vmax=None, annotate=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(4, 0.35 * len(col_labels) + 2), max(3, 0.35 * len(row_labels) + 1.5)))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if (M[i, j] - (vmin or M.min())) < 0.5 * ((vmax or M.max()) - (vmin or M.min())) else "black")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_lines(series: dict, xlabel, ylabel, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name, ys in series.items():
        ax.plot(range(1, len(ys) + 1), ys, marker="o", ms=3, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_hist(values: np.ndarray, xlabel, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(values, bins=40)
    ax.set_xlabel(xlabel); ax.set_ylabel("units"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device}")
    md = []  # report lines

    def section(title):
        md.append(f"\n## {title}\n")

    def table(header, rows):
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "---|" * len(header))
        for r in rows:
            md.append("| " + " | ".join(str(c) if not isinstance(c, float) else f"{c:.3f}" for c in r) + " |")
        md.append("")

    images, y_coarse, y_fine, groups = load_images(args, rng)
    model = build_model(args, device)
    probe = StmProbe(model)
    H = model.model_actor.get_model(0).config.hidden_size if probe.sparse else model.model_actor.config.hidden_size

    dump_path = os.path.join(args.out_dir, "dump.npz")
    if args.reuse_dump and os.path.exists(dump_path):
        d = dict(np.load(dump_path))
        logger.info(f"Reused {dump_path}")
        for k, v in (("y_coarse", y_coarse), ("y_fine", y_fine), ("groups", groups)):
            assert d[k].shape == v.shape and (d[k] == v).all(), \
                f"{k} in {dump_path} differs from the reloaded images: use the dump's --seed/--max-per-label/--fine-groups"
    else:
        d = run_episodes(model, probe, images, args.steps, args.batch_size, device)
        d.update(y_coarse=y_coarse, y_fine=y_fine, groups=groups)
        np.savez_compressed(dump_path, **d)
        logger.info(f"Wrote {dump_path}")

    T = d["logits"].shape[1]
    N = len(y_fine)
    fine_labs = np.unique(y_fine)
    fine_names = [f"c{fine_label_to_coarse(l, y_fine, y_coarse)}/g{fine_label_to_group()[l]}/f{l}" for l in fine_labs]
    pred_t = d["logits"].argmax(-1)  # [N, T]
    pred_0 = d["logits0"].argmax(-1)

    md.append(f"# STM inspectability report\n")
    md.append(f"LTM `{args.ltm_checkpoint}`, STM `{args.stm_checkpoint}`, sparsity {args.sparsity}, "
              f"{N} {args.split} images, coarse {args.coarse_classes}, groups {args.fine_groups}, {T} steps, mean bias.\n")

    # ---- accuracy per step (sanity + B5 accuracy trajectory) ----
    section("Accuracy over the episode (sanity check against the evaluate branch)")
    acc_steps = [float((pred_t[:, t] == y_coarse).mean()) for t in range(T)]
    rows = [["unbiased (step 0)"] + [f"{v:.3f}" for v in accuracy_by(pred_0, y_coarse, groups).values()] + [f"{(pred_0 == y_coarse).mean():.3f}"]]
    for t in range(T):
        rows.append([f"step {t + 1}"] + [f"{v:.3f}" for v in accuracy_by(pred_t[:, t], y_coarse, groups).values()] + [f"{acc_steps[t]:.3f}"])
    table(["", *[f"group {g}" for g in np.unique(groups)], "all"], rows)

    # ---- B5 bias trajectory ----
    section("B5 Bias trajectory over the episode")
    b = d["bias"]  # [N, T, 128]
    b_final = b[:, -1]
    cos_to_final = [float(np.mean(np.sum(b[:, t] * b_final, 1) / np.maximum(np.linalg.norm(b[:, t], axis=1) * np.linalg.norm(b_final, axis=1), 1e-12))) for t in range(T)]
    mag = [float(np.abs(b[:, t]).sum(1).mean()) for t in range(T)]
    between_fine = [variance_between_fraction(b[:, t], y_fine) for t in range(T)]
    between_coarse = [variance_between_fraction(b[:, t], y_coarse) for t in range(T)]
    table(["step", "|bias| sum", "cos(b_t, b_T)", "var. between fine labels", "var. between coarse", "accuracy"],
          [[t + 1, mag[t], cos_to_final[t], between_fine[t], between_coarse[t], acc_steps[t]] for t in range(T)])
    md.append("Variance fractions: how much of the bias variance across images is explained by the label means "
              "(1 = a pure per-class template, 0 = image-specific).\n")
    plot_lines({"cos(b_t, b_T)": cos_to_final, "between-fine-label var. fraction": between_fine,
                "between-coarse var. fraction": between_coarse, "accuracy": acc_steps},
               "episode step", "", "Bias trajectory", os.path.join(args.out_dir, "b5_trajectory.png"))

    # ---- B1 templates ----
    section("B1 Per-class bias templates (mean final-step bias)")
    templates = np.stack([b_final[y_fine == l].mean(0) for l in fine_labs])  # [F, 128]
    order = np.argsort(templates.mean(0))  # sort channels by mean gating for readability
    plot_heatmap(templates[:, order], fine_names, [str(c) for c in order], "Mean final bias per fine label (channels sorted)",
                 os.path.join(args.out_dir, "b1_templates.png"), cmap="RdBu_r", vmin=-1, vmax=1)
    sim = cosine_matrix(templates)
    plot_heatmap(sim, fine_names, fine_names, "Cosine similarity of class templates",
                 os.path.join(args.out_dir, "b1_template_similarity.png"), vmin=-1, vmax=1, annotate=True)
    same_c = np.array([[fine_label_to_coarse(a, y_fine, y_coarse) == fine_label_to_coarse(bb, y_fine, y_coarse) for bb in fine_labs] for a in fine_labs])
    off = ~np.eye(len(fine_labs), dtype=bool)
    table(["", "mean cosine"], [["templates, same coarse class", float(sim[same_c & off].mean())],
                                ["templates, different coarse class", float(sim[~same_c].mean())],
                                ["channels suppressed (< -0.5) on average", int((templates.mean(0) < -0.5).sum())],
                                ["channels amplified (> 0.5) on average", int((templates.mean(0) > 0.5).sum())]])
    md.append("Reading: if templates cluster by coarse class the bias encodes the *label* (steering); if they cluster by "
              "fine label within a coarse class it encodes the *domain*; if all are alike it is one global re-weighting.\n")

    # ---- B4 movement toward prototypes ----
    section("B4 Movement of the LTM features toward class prototypes")
    P, plabs = ncm_prototypes(d["feat0"], y_coarse)  # unbiased prototypes of the coarse classes
    own = np.searchsorted(plabs, y_coarse)
    cos0 = cosine_to_prototypes(d["feat0"], P); cosT = cosine_to_prototypes(d["feat"][:, -1], P)
    rows = []
    for name, C in [("unbiased", cos0), ("final bias", cosT)]:
        own_cos = C[np.arange(N), own]
        other = C.copy(); other[np.arange(N), own] = -np.inf
        rows.append([name, float(own_cos.mean()), float(other.max(1).mean()), float((C.argmax(1) == own).mean())])
    table(["features", "cos to own prototype", "cos to nearest other", "NCM accuracy (unbiased prototypes)"], rows)
    md.append("If the STM's gain moves samples toward their own (unbiased) prototype, its action can be read as "
              "'make this image look more like a typical member of its class' in the LTM's own feature space.\n")

    # ---- A1 mask overlap ----
    if probe.sparse:
        section("A1 Overlap of the actor's active hidden units")
        rows = []
        for t in [0, T - 1]:
            stats, sets = jaccard_stats(d["mask"][:, t], groups, y_fine, H, rng)
            rows.append([f"step {t + 1}"] + [stats[k] for k in stats])
        table(["", *stats.keys()], rows)
        _, sets_T = jaccard_stats(d["mask"][:, -1], groups, y_fine, H, rng)
        U, ulabs = unit_usage(sets_T, y_fine)  # [F, H]
        plot_heatmap(cosine_matrix(U), fine_names, fine_names, "Cosine similarity of unit-usage profiles (final step)",
                     os.path.join(args.out_dir, "a1_usage_similarity.png"), vmin=0, vmax=1, annotate=True)
        md.append("Jaccard is over the k active units of two images. The last column is the expectation for two random "
                  "k-of-H sets. Same-label overlap far above it means the fixed random projection already routes a class "
                  "to its own units; the gap to different-group overlap is the separation available to the STM.\n")
        active_sets = sets_T
    else:
        # dense ablation: define 'active' as the top-k units by magnitude of the first hidden layer
        k = 32
        h1 = d["hidden"][:, -1, 0].astype(np.float32)
        idx = np.argsort(-np.abs(h1), axis=1)[:, :k]
        active_sets = np.zeros((N, H), dtype=bool); np.put_along_axis(active_sets, idx, True, axis=1)
        U, ulabs = unit_usage(active_sets, y_fine)
        section("A1 (dense ablation) top-32 units by magnitude used as the 'active set'")

    # ---- A2 selectivity ----
    section("A2 Class selectivity of hidden units")
    sel = selectivity(U)
    table(["", "value"], [[k, v] for k, v in sel.items() if k != "entropies"])
    plot_hist(sel["entropies"], "normalised entropy of p(fine label | unit active)", "Unit selectivity (0 = one class)",
              os.path.join(args.out_dir, "a2_selectivity.png"))
    # activation-weighted version: mean |h| per (label, unit) in the first hidden layer
    h1 = d["hidden"][:, -1, 0].astype(np.float32)
    Uw = np.stack([np.abs(h1[y_fine == l]).mean(0) for l in fine_labs])
    selw = selectivity(Uw)
    md.append(f"Activation-weighted (mean |h| of hidden layer 1): median normalised entropy {selw['median normalised entropy']:.3f}, "
              f"fraction < 0.5: {selw['fraction entropy < 0.5']:.3f}.\n")

    # ---- A5 decodability ----
    section("A5 Linear decodability (5-fold logistic regression)")
    feats = {
        "LTM stage-2 pooled, unbiased (STM input)": d["e0"],
        "LTM 512-d features, unbiased": d["feat0"],
        "LTM 512-d features, final bias": d["feat"][:, -1],
        "STM hidden layer 1, final step": d["hidden"][:, -1, 0].astype(np.float32),
        "STM hidden layer 2, final step": d["hidden"][:, -1, 1].astype(np.float32),
        "STM final bias (128)": b_final,
    }
    rows = []
    for name, X in feats.items():
        rows.append([name, linear_probe(X, y_fine, args.seed), linear_probe(X, groups, args.seed), linear_probe(X, y_coarse, args.seed)])
    table(["representation", f"fine label ({len(fine_labs)}-way)", f"group ({len(np.unique(groups))}-way)", "coarse (2-way)"], rows)
    md.append("The STM is not trained to identify fine labels; if its hidden code decodes them better than the features it "
              "reads, it has separated the domains on its own. Note the hidden layers are sparse (k active units).\n")

    # ---- A3 checkpoint comparison ----
    if args.compare_checkpoints:
        section("A3 Which hidden units changed between checkpoints")
        md.append("Per hidden unit of the actor: L2 norm of the change of its incoming (layer-1 rows) and outgoing (layer-2 "
                  "columns) weights between successive checkpoints, against the usage of that unit by each fine-class group "
                  "in this dump.\n")
        sds = [torch.load(p, weights_only=True, map_location="cpu")["model_actor"] for p in args.compare_checkpoints]
        keys = [k for k in sds[0] if k.endswith("dense-layer-0.weight") or k.endswith("dense-layer-1.weight")]
        w0_key = [k for k in keys if "layer-0" in k][0]; w1_key = [k for k in keys if "layer-1" in k][0]
        Ug, glabs = unit_usage(active_sets, groups)
        rows = []
        for a, bb in zip(sds[:-1], sds[1:]):
            d_in = (bb[w0_key] - a[w0_key]).norm(dim=1).numpy()   # [H]: incoming to unit
            d_out = (bb[w1_key] - a[w1_key]).norm(dim=0).numpy()  # [H]: outgoing from unit (layer-2 input column)
            change = d_in + d_out
            touched = change > 1e-6 * max(change.max(), 1e-12)
            top = np.argsort(-change)[:args.lesion_n]
            corr = [float(np.corrcoef(change, Ug[i])[0, 1]) for i in range(len(glabs))]
            rows.append([f"{int(touched.sum())}/{H}", f"{change[top].sum() / max(change.sum(), 1e-12):.2f}"] + [f"{c:.2f}" for c in corr])
        table(["units changed", f"share of change in top {args.lesion_n}", *[f"corr(change, usage g{g})" for g in glabs]], rows)
        md.append("Rows are successive checkpoint pairs in the order given. A high correlation with the usage of the group "
                  "trained in that phase, and a low one with the others, is the localized-update result.\n")

    # ---- A4 lesion (model in the loop) ----
    if args.lesion:
        section(f"A4 Lesion: remove the {args.lesion_n} units most selective for each group, re-run the episode")
        Ug, glabs = unit_usage(active_sets, groups)  # [G, H]
        rows = []
        base = accuracy_by(pred_t[:, -1], y_coarse, groups)
        rows.append(["none"] + [f"{base[g]:.3f}" for g in glabs] + [f"{acc_steps[-1]:.3f}"])
        matrix = []
        for gi, g in enumerate(glabs):
            others = np.delete(Ug, gi, axis=0).max(0) if len(glabs) > 1 else 0
            score = Ug[gi] - others
            lesion_units = np.argsort(-score)[:args.lesion_n]
            lesioned = StmProbe(model, lesion_units=lesion_units)
            dl = run_episodes(model, lesioned, images, args.steps, args.batch_size, device, record=False)
            pl = dl["logits"][:, -1].argmax(-1)
            acc = accuracy_by(pl, y_coarse, groups)
            matrix.append([acc[h] - base[h] for h in glabs])
            rows.append([f"group {g}"] + [f"{acc[h]:.3f}" for h in glabs] + [f"{(pl == y_coarse).mean():.3f}"])
        table(["lesioned units of", *[f"acc. group {g}" for g in glabs], "all"], rows)
        plot_heatmap(np.array(matrix), [f"lesion g{g}" for g in glabs], [f"eval g{g}" for g in glabs],
                     "Accuracy change after lesion", os.path.join(args.out_dir, "a4_lesion.png"), cmap="RdBu_r", vmin=-0.3, vmax=0.3, annotate=True)
        md.append("A diagonal (own-group accuracy falls, others hold) is the causal version of separability. The random "
                  f"control below is the expected loss from removing any {args.lesion_n} units.\n")
        # random-lesion control
        rand_units = rng.choice(H, size=args.lesion_n, replace=False)
        dl = run_episodes(model, StmProbe(model, lesion_units=rand_units), images, args.steps, args.batch_size, device, record=False)
        pl = dl["logits"][:, -1].argmax(-1); acc = accuracy_by(pl, y_coarse, groups)
        md.append("Random-lesion control: " + ", ".join(f"g{g} {acc[g]:.3f}" for g in glabs) + f", all {(pl == y_coarse).mean():.3f}\n")

    # ---- B2 / B3 templates as fixed gates (model in the loop) ----
    if args.templates:
        section("B2 Template sufficiency and B3 steering (class-mean bias as a fixed gate, no STM)")
        rows = []
        acc_unb = accuracy_by(pred_0, y_coarse, groups)
        acc_stm = accuracy_by(pred_t[:, -1], y_coarse, groups)
        # own template
        own_t = templates[np.searchsorted(fine_labs, y_fine)]
        pred_own = []
        for start in range(0, N, args.batch_size):
            img = images[start:start + args.batch_size].to(device)
            lg, _, _ = ltm_pass(model, img, torch.as_tensor(own_t[start:start + args.batch_size], device=device))
            pred_own.append(lg.argmax(-1).cpu().numpy())
        pred_own = np.concatenate(pred_own); acc_own = accuracy_by(pred_own, y_coarse, groups)
        # global template
        glob = b_final.mean(0, keepdims=True)
        pred_glob = []
        for start in range(0, N, args.batch_size):
            img = images[start:start + args.batch_size].to(device)
            lg, _, _ = ltm_pass(model, img, torch.as_tensor(np.repeat(glob, img.shape[0], 0), device=device))
            pred_glob.append(lg.argmax(-1).cpu().numpy())
        pred_glob = np.concatenate(pred_glob); acc_glob = accuracy_by(pred_glob, y_coarse, groups)
        glabs = np.unique(groups)
        for name, acc, pr in [("unbiased LTM", acc_unb, pred_0), ("one global template", acc_glob, pred_glob),
                              ("own-class template (oracle)", acc_own, pred_own), ("full STM episode", acc_stm, pred_t[:, -1])]:
            rows.append([name] + [f"{acc[g]:.3f}" for g in glabs] + [f"{(pr == y_coarse).mean():.3f}"])
        table(["gate", *[f"group {g}" for g in glabs], "all"], rows)
        md.append("If the own-class template recovers most of the STM's gain, the STM's decision on any image can be "
                  "explained as 'apply this class's channel recipe'; the residual is what the per-image episode adds.\n")
        # B3: every template on every class -> mean probability assigned to the template's coarse class
        tcoarse = np.array([fine_label_to_coarse(l, y_fine, y_coarse) for l in fine_labs])
        steer = np.zeros((len(fine_labs), len(fine_labs)))  # [image label, template]
        acc_cross = np.zeros_like(steer)
        for j, tmpl in enumerate(templates):
            probs = []
            for start in range(0, N, args.batch_size):
                img = images[start:start + args.batch_size].to(device)
                lg, _, _ = ltm_pass(model, img, torch.as_tensor(np.repeat(tmpl[None], img.shape[0], 0), device=device))
                probs.append(F.softmax(lg, -1).cpu().numpy())
            probs = np.concatenate(probs)
            for i, l in enumerate(fine_labs):
                sel_i = y_fine == l
                steer[i, j] = probs[sel_i, tcoarse[j]].mean()
                acc_cross[i, j] = (probs[sel_i].argmax(1) == y_coarse[sel_i]).mean()
        p0 = F.softmax(torch.as_tensor(d["logits0"]), -1).numpy()
        base_p = np.array([[p0[y_fine == l, tcoarse[j]].mean() for j in range(len(fine_labs))] for l in fine_labs])
        plot_heatmap(steer - base_p, fine_names, fine_names, "B3 steering: change in P(template's coarse class) vs unbiased",
                     os.path.join(args.out_dir, "b3_steering.png"), cmap="RdBu_r", vmin=-0.5, vmax=0.5, annotate=True)
        plot_heatmap(acc_cross, fine_names, fine_names, "Accuracy of images (rows) under each template (columns)",
                     os.path.join(args.out_dir, "b3_cross_accuracy.png"), vmin=0, vmax=1, annotate=True)
        cross_c = ~same_c
        md.append(f"Steering: applying a template of the *other* coarse class raises P(that class) by "
                  f"{float((steer - base_p)[cross_c].mean()):.3f} on average (own-class templates: "
                  f"{float((steer - base_p)[same_c].mean()):.3f}). Positive cross-class steering means the bias carries the label.\n")

    with open(os.path.join(args.out_dir, "report.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Report: {os.path.join(args.out_dir, 'report.md')}")


def fine_label_to_coarse(fine_label: int, y_fine: np.ndarray, y_coarse: np.ndarray) -> int:
    return int(y_coarse[y_fine == fine_label][0])


if __name__ == "__main__":
    main()
