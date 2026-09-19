"""
Continual-learning baseline "heads" on top of the frozen, pre-trained ResNet-18 LTM.

Replicates the protocol of cifar_main_ltm_fine_tuning.py / cifar_main_stm_training.py
(coarse-class pair, sequential fine-class phases, evaluation after every epoch on the
"12", "3", "4", "5" test subsets, CifarResults output format) but instead of fine-tuning
the whole LTM, or training the RL-controlled STM, it trains one of four cheap heads on
the frozen 512-d pooled encoder features:

    linear    nn.Linear(512, 20), initialised from the LTM's own classifier, SGD + CE.
    ncm       Nearest-class-mean prototypes (cosine), purely online running means.
    flymodel  Shen, Dasgupta & Navlakha 2021: sparse random projection + top-k WTA +
              Hebbian, clipped output weights (one-pass, no gradient).
    sdmlp     Bricken et al. 2023 "Sparse Distributed Memory is a Continual Learner":
              L2-normalised positive addresses, Top-K with GABA switch, SGD + CE.

The encoder is run exactly once per (checkpoint, coarse-class pair) and the 512-d encodings
are cached to disk, so every method/seed/order re-run is instant.

Example:
    python cifar_main_head_baselines.py --method sdmlp --experiment-type continual \
        --fine-classes 3 4 5 --coarse-classes 0 1 --seed 0 \
        --checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from environment.cifar.cifar_classifier import CifarClassifier
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_results import CifarResults
from model.resnet import ResNetConfig
from util.instrumentation import Instrumentation
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 20          # coarse classes; all heads emit 20-way logits
ENCODING_DIM = 512        # ResNet-18 pooled feature size
EVALUATE_NAMES = ["12", "3", "4", "5"]
TARGET_STEPS = 6250       # from cifar_main_ltm_fine_tuning.py: NUM_EPOCHS = int(6250 / instances_per_epoch)

EXPERIMENT_TYPE_CONTINUAL = "continual"
EXPERIMENT_TYPE_STREAMING = "streaming"

METHOD_DEFAULT_LR = {"linear": 0.01, "ncm": 0.0, "flymodel": 0.2, "sdmlp": 0.05}
METHOD_DEFAULT_MOMENTUM = {"linear": 0.0, "ncm": 0.0, "flymodel": 0.0, "sdmlp": 0.9}


# --------------------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["linear", "ncm", "flymodel", "sdmlp"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default="../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth",
                   help="Frozen LTM state_dict (CifarClassifier(ResNetConfig(num_classes=20), bias_stage=-1)).")
    p.add_argument("--data-path", type=str, default="../cifar-100-python")
    p.add_argument("--experiment-type", choices=[EXPERIMENT_TYPE_CONTINUAL, EXPERIMENT_TYPE_STREAMING],
                   default=EXPERIMENT_TYPE_CONTINUAL)
    p.add_argument("--batch-size", type=int, default=None, help="Default: 16 (continual) or 1 (streaming).")
    p.add_argument("--max-instances", type=int, default=None,
                   help="Training instances per coarse class per phase (None = all 500, as the reference).")
    p.add_argument("--fine-classes", type=int, nargs="+", default=[3, 4, 5],
                   help="Sequence of fine-class phases, e.g. 3 4 5 (continual) or a single class (streaming).")
    p.add_argument("--coarse-classes", type=int, nargs="+", default=[0, 1])
    p.add_argument("--pretrain-epochs", type=int, default=12,
                   help="Epochs of head training on fine-classes 1,2 of the coarse pair before the continual phases "
                        "(mirrors the STM pretraining). 0 disables.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Epochs per phase. Default: int(6250 / instances_per_epoch) = 12 at 500 instances, "
                        "exactly as cifar_main_ltm_fine_tuning.py (independent of batch size).")
    p.add_argument("--lr", type=float, default=None,
                   help=f"Learning rate. Default per method: {METHOD_DEFAULT_LR}")
    p.add_argument("--momentum", type=float, default=None,
                   help=f"SGD momentum (linear, sdmlp). Default per method: {METHOD_DEFAULT_MOMENTUM}")
    # flymodel
    p.add_argument("--n-kc", type=int, default=20000, help="FlyModel: number of Kenyon cells.")
    p.add_argument("--n-response", type=int, default=64, help="FlyModel: ones per row of the random projection.")
    p.add_argument("--k-frac", type=float, default=0.01, help="FlyModel: fraction of KCs kept by the WTA.")
    # sdmlp
    p.add_argument("--nneurons", type=int, default=1000, help="SDMLP: hidden neurons (paper STM h=1000).")
    p.add_argument("--k", type=int, default=32, help="SDMLP: Top-K k_min (paper STM sparsity 32).")
    p.add_argument("--gaba-switch-activations", type=int, default=None,
                   help="SDMLP: binary activations per neuron for the GABA switch to complete. "
                        "Default: half the number of pretraining samples (so the switch completes during "
                        "pretraining), or 1 if --pretrain-epochs 0.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="SDMLP: gradient-norm clip (Bricken default 1.0).")
    # plumbing
    p.add_argument("--run-path", type=str, default=None,
                   help="Explicit output directory (default: timestamped runs/cifar_100/<experiment name>/...).")
    p.add_argument("--runs-root", type=str, default="./runs")
    p.add_argument("--cache-dir", type=str, default="./runs_local/encodings")
    p.add_argument("--device", type=str, default=None, help="Encoder device (default: mps if available else cpu).")
    p.add_argument("--head-device", type=str, default="cpu", help="Device for the head (cpu is plenty).")
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--write-random-checkpoint", type=str, default=None,
                   help="Dev helper: write a randomly initialised LTM state_dict to this path and exit.")
    args = p.parse_args()

    if args.batch_size is None:
        args.batch_size = 1 if args.experiment_type == EXPERIMENT_TYPE_STREAMING else 16
    if args.lr is None:
        args.lr = METHOD_DEFAULT_LR[args.method]
    if args.momentum is None:
        args.momentum = METHOD_DEFAULT_MOMENTUM[args.method]
    return args


# --------------------------------------------------------------------------------------
# Frozen LTM + encodings cache
# --------------------------------------------------------------------------------------
def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_ltm(checkpoint: str, device: torch.device) -> CifarClassifier:
    config = ResNetConfig(num_classes=NUM_CLASSES)
    model = CifarClassifier(config, bias_stage=-1)
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # BatchNorm in inference mode, exactly as the STM uses the classifier
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def write_random_checkpoint(path: str, seed: int):
    torch.manual_seed(seed)
    model = CifarClassifier(ResNetConfig(num_classes=NUM_CLASSES), bias_stage=-1)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Wrote random LTM checkpoint to {path}")


def _release_shared_memory(dataset: Cifar100Dataset):
    """Cifar100Dataset keeps its arrays in multiprocessing shared memory and never frees them."""
    for shm in (dataset.shared_memory_images, dataset.shared_memory_labels_coarse, dataset.shared_memory_labels_fine):
        if shm is None:
            continue
        try:
            shm.close()
            shm.unlink()
        except Exception:  # noqa: BLE001 - best effort
            pass


def build_dataset(data_path: str, training: bool, coarse_classes: list[int], fine_class_group: list[int]):
    """Same construction as the reference script (label_type coarse, exclusion sets), all instances."""
    all_groups = [1, 2, 3, 4, 5]
    exclude_fine = Cifar100Dataset.get_fine_classes([g for g in all_groups if g not in fine_class_group])
    exclude_coarse = Cifar100Dataset.get_coarse_classes_excluded(coarse_classes)
    return Cifar100Dataset(
        file_path=data_path,
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=training,
        exclude_classes_coarse=exclude_coarse,
        exclude_classes_fine=exclude_fine,
        max_instances=None,
        as_tensor=True,
    )


@torch.no_grad()
def encode_dataset(model: CifarClassifier, dataset: Cifar100Dataset, device: torch.device, batch_size: int):
    feats = []
    n = len(dataset)
    for start in range(0, n, batch_size):
        x = dataset.images[start:start + batch_size].to(device)
        f, _ = model.encoder(x, bias=None)  # [B, 512] pooled features, what model.classifier consumes
        feats.append(f.float().cpu())
    X = torch.cat(feats, dim=0).numpy() if feats else np.zeros((0, ENCODING_DIM), np.float32)
    y_coarse = np.asarray(dataset.labels_coarse, dtype=np.int64)
    y_fine = np.asarray(dataset.labels_fine, dtype=np.int64)
    return X, y_coarse, y_fine


GROUPS = {"12": [1, 2], "3": [3], "4": [4], "5": [5]}


def get_encodings(args, device: torch.device) -> dict:
    """
    Returns dict with keys f"{split}_{name}" -> (X, y_coarse, y_fine) for split in {train, test},
    name in {"12","3","4","5"}, restricted to the coarse pair. Cached on disk keyed by checkpoint
    hash + coarse classes.
    """
    ckpt_hash = file_hash(args.checkpoint)
    cc = "-".join(str(c) for c in sorted(args.coarse_classes))
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"encodings_{ckpt_hash}_cc{cc}.npz")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached encodings: {cache_path}")
        z = np.load(cache_path)
        out = {}
        for split in ("train", "test"):
            for name in GROUPS:
                key = f"{split}_{name}"
                out[key] = (z[f"X_{key}"], z[f"yc_{key}"], z[f"yf_{key}"])
        return out

    logger.info(f"Encoding datasets with frozen LTM {args.checkpoint} (hash {ckpt_hash}) on {device}...")
    model = load_ltm(args.checkpoint, device)
    out, arrays = {}, {}
    t0 = time.time()
    for split in ("train", "test"):
        for name, group in GROUPS.items():
            ds = build_dataset(args.data_path, split == "train", args.coarse_classes, group)
            X, yc, yf = encode_dataset(model, ds, device, args.encode_batch_size)
            _release_shared_memory(ds)
            key = f"{split}_{name}"
            out[key] = (X, yc, yf)
            arrays[f"X_{key}"], arrays[f"yc_{key}"], arrays[f"yf_{key}"] = X, yc, yf
            logger.info(f"  {key}: {X.shape[0]} instances, coarse labels {sorted(set(yc.tolist()))}")
    logger.info(f"Encoding took {time.time() - t0:.1f}s; caching to {cache_path}")
    # Also stash the LTM classifier so the linear head can be initialised without reloading the LTM
    arrays["ltm_classifier_weight"] = model.classifier.linear.weight.detach().cpu().numpy()
    arrays["ltm_classifier_bias"] = model.classifier.linear.bias.detach().cpu().numpy()
    np.savez(cache_path, **arrays)
    return out


def load_ltm_classifier(args) -> tuple[np.ndarray, np.ndarray]:
    ckpt_hash = file_hash(args.checkpoint)
    cc = "-".join(str(c) for c in sorted(args.coarse_classes))
    cache_path = os.path.join(args.cache_dir, f"encodings_{ckpt_hash}_cc{cc}.npz")
    z = np.load(cache_path)
    return z["ltm_classifier_weight"], z["ltm_classifier_bias"]


def subsample_per_coarse_class(X, yc, yf, coarse_classes: list[int], max_instances: int | None, rng: np.random.Generator):
    """Mirror Cifar100Dataset.sample_data: max_instances chosen without replacement per coarse class."""
    if max_instances is None:
        return X, yc, yf
    idx = []
    for c in sorted(coarse_classes):
        c_idx = np.flatnonzero(yc == c)
        if len(c_idx) < max_instances:
            raise ValueError(f"Coarse class {c} only has {len(c_idx)} instances, max_instances={max_instances}")
        idx.extend(rng.choice(c_idx, size=max_instances, replace=False).tolist())
    idx = np.asarray(idx, dtype=np.int64)
    return X[idx], yc[idx], yf[idx]


# --------------------------------------------------------------------------------------
# Heads. All operate on [B, 512] float tensors and emit [B, 20] logits.
# --------------------------------------------------------------------------------------
class Head:
    name = "head"

    def fit_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """One online update. Returns the pre-update logits (for training accuracy, as the reference)."""
        raise NotImplementedError

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class LinearHead(Head):
    """Linear probe on the frozen features, initialised from the LTM's own classifier weights."""
    name = "linear"

    def __init__(self, init_weight: np.ndarray, init_bias: np.ndarray, lr: float, momentum: float, device):
        self.linear = nn.Linear(ENCODING_DIM, NUM_CLASSES).to(device)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(init_weight))
            self.linear.bias.copy_(torch.from_numpy(init_bias))
        self.opt = torch.optim.SGD(self.linear.parameters(), lr=lr, momentum=momentum)

    def fit_batch(self, x, y):
        self.opt.zero_grad()
        logits = self.linear(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        return logits.detach()

    @torch.no_grad()
    def logits(self, x):
        return self.linear(x)


class NCMHead(Head):
    """Nearest class mean with cosine similarity. 20 prototype slots; unseen slots score -inf."""
    name = "ncm"

    def __init__(self, device):
        self.proto = torch.zeros(NUM_CLASSES, ENCODING_DIM, device=device)
        self.counts = torch.zeros(NUM_CLASSES, device=device)

    def fit_batch(self, x, y):
        logits = self.logits(x)
        for c in torch.unique(y):
            xs = x[y == c]
            n = xs.shape[0]
            self.proto[c] = (self.proto[c] * self.counts[c] + xs.sum(0)) / (self.counts[c] + n)
            self.counts[c] += n
        return logits

    @torch.no_grad()
    def logits(self, x):
        xn = F.normalize(x, dim=1)
        pn = F.normalize(self.proto, dim=1)
        sims = xn @ pn.T  # = 1 - cosine distance
        sims[:, self.counts == 0] = float("-inf")
        return sims


class FlyModelHead(Head):
    """
    Shen, Dasgupta & Navlakha 2021 (fly_model_cifar100_with_errorbars.ipynb):
      x -> min-max scale per row -> KC = R x (R binary, n_response ones per row) -> ReLU
        -> keep the top k_frac (quantile 1-k_frac, zero the rest) -> divide by row max.
      W[c] += lr * sum_{i: y_i=c} KC_i ; W = min(W, 1).   predict argmax_c W[c].KC
    Hebbian, one-pass; the per-batch clipped sum is identical to the notebook's per-class clipped sum
    (increments are non-negative so clipping commutes with summation).
    """
    name = "flymodel"

    def __init__(self, n_kc: int, n_response: int, k_frac: float, lr: float, seed: int, device):
        self.n_kc, self.k_frac, self.lr = n_kc, k_frac, lr
        g = torch.Generator().manual_seed(seed)
        # n_response distinct ones per row (vectorised equivalent of random.sample per row)
        cols = torch.rand(n_kc, ENCODING_DIM, generator=g).argsort(dim=1)[:, :n_response]
        R = torch.zeros(n_kc, ENCODING_DIM).scatter_(1, cols, 1.0)
        self.R = R.to(device)
        self.W = torch.zeros(NUM_CLASSES, n_kc, device=device)

    @torch.no_grad()
    def kc(self, x):
        mn = x.min(dim=1, keepdim=True).values
        mx = x.max(dim=1, keepdim=True).values
        scale = torch.where(mx - mn > 0, mx - mn, torch.ones_like(mx))  # sklearn minmax_scale zero-range guard
        x = (x - mn) / scale
        kc = x @ self.R.T
        kc = torch.clamp(kc, min=0)  # KC[KC <= thresh(=0)] = 0
        thr = torch.quantile(kc, 1.0 - self.k_frac, dim=1, keepdim=True)
        kc = torch.where(kc < thr, torch.zeros_like(kc), kc)
        mx = kc.max(dim=1, keepdim=True).values
        kc = kc / torch.where(mx > 0, mx, torch.ones_like(mx))
        return kc

    @torch.no_grad()
    def fit_batch(self, x, y):
        kc = self.kc(x)
        logits = kc @ self.W.T
        for c in torch.unique(y):
            self.W[c] += self.lr * kc[y == c].sum(0)
        self.W.clamp_(max=1.0)
        return logits

    @torch.no_grad()
    def logits(self, x):
        return self.kc(x) @ self.W.T


class SDMLPHead(Head):
    """
    Self-contained port of SDMContinualLearner models/SDM_Base.py + models/TopK_Act.py with the paper's
    CIFAR settings (k_approach "GABA_SWITCH_ACT_BIN", use_bias False, all_positive_weights True,
    norm_addresses True, norm_values False, SGDM, gradient clip 1.0):

      x -> ReLU -> L2-normalise rows -> fc1 (no bias; rows L2-normalised, all >= 0)
        -> Top-K: a = ReLU(a); inhib = min of the top (k+1) values per sample;
                  gaba = clamp(-1 + 2 * counter / N_switch, -1, 1)  (per neuron)
                  a = ReLU(a - gaba * inhib);  counter += (a > 0) summed over the batch
        -> purkinje (no bias) -> 20 logits.
      After every SGD step: clamp ALL weights >= 0 (Bricken's enforce_positive_weights applies to every
      parameter, purkinje included) and re-normalise fc1 rows (enforce_l2_norm_weights).
    Deviation: activation counters are only incremented in training mode (the original increments during
    validation passes too, which is an artefact of running Top_K.forward under Lightning validation).
    """
    name = "sdmlp"

    def __init__(self, nneurons: int, k: int, gaba_switch_activations: int, lr: float, momentum: float,
                 grad_clip: float, device):
        self.k = k
        self.gaba_switch_activations = max(1, int(gaba_switch_activations))
        self.grad_clip = grad_clip
        self.fc1 = nn.Linear(ENCODING_DIM, nneurons, bias=False).to(device)
        self.purkinje = nn.Linear(nneurons, NUM_CLASSES, bias=False).to(device)
        self.counters = torch.zeros(1, nneurons, device=device)
        self.params = list(self.fc1.parameters()) + list(self.purkinje.parameters())
        self.opt = torch.optim.SGD(self.params, lr=lr, momentum=momentum)
        self.enforce_constraints()  # on_fit_start

    @torch.no_grad()
    def enforce_constraints(self):
        for p in self.params:
            p.data.clamp_(0)
        self.fc1.weight.data /= torch.norm(self.fc1.weight.data, dim=1, keepdim=True).clamp_min(1e-12)

    def gaba_response(self):
        lin = 2.0 / self.gaba_switch_activations
        return torch.clamp(-1.0 + lin * self.counters, -1.0, 1.0)

    def top_k(self, a, training: bool):
        a = F.relu(a)
        kk = min(self.k + 1, a.shape[1])  # +1: GABA switch subtracts the (k+1)-th value
        vals, _ = torch.topk(a, kk, dim=1, sorted=False)
        inhib = vals.detach().min(dim=1, keepdim=True).values
        a = F.relu(a - self.gaba_response() * inhib)
        if training:
            self.counters += (a > 0).sum(dim=0, keepdim=True)
        return a

    def forward(self, x, training: bool):
        x = F.relu(x)
        x = x / torch.norm(x, dim=1, keepdim=True).clamp_min(1e-12)
        a = self.top_k(self.fc1(x), training)
        return self.purkinje(a)

    def fit_batch(self, x, y):
        self.opt.zero_grad()
        logits = self.forward(x, training=True)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.opt.step()
        self.enforce_constraints()  # on_train_batch_end
        return logits.detach()

    @torch.no_grad()
    def logits(self, x):
        return self.forward(x, training=False)

    def describe(self):
        switched = float((self.counters >= self.gaba_switch_activations).float().mean())
        return f"sdmlp k={self.k} N_switch={self.gaba_switch_activations} frac. neurons switched={switched:.2f}"


def make_head(args, device) -> Head:
    if args.method == "linear":
        w, b = load_ltm_classifier(args)
        return LinearHead(w, b, lr=args.lr, momentum=args.momentum, device=device)
    if args.method == "ncm":
        return NCMHead(device)
    if args.method == "flymodel":
        return FlyModelHead(args.n_kc, args.n_response, args.k_frac, args.lr, args.seed, device)
    if args.method == "sdmlp":
        return SDMLPHead(args.nneurons, args.k, args.gaba_switch_activations, args.lr, args.momentum,
                         args.grad_clip, device)
    raise ValueError(args.method)


# --------------------------------------------------------------------------------------
# Training / evaluation loop
# --------------------------------------------------------------------------------------
@dataclass
class EpochTiming:
    train_s: float = 0.0
    eval_s: float = 0.0


@torch.no_grad()
def evaluate(head: Head, X: torch.Tensor, y: torch.Tensor, chunk: int = 1024) -> float:
    correct = 0
    for s in range(0, X.shape[0], chunk):
        correct += (head.logits(X[s:s + chunk]).argmax(dim=1) == y[s:s + chunk]).sum().item()
    return correct / max(1, X.shape[0])


def train_epoch(head: Head, X: torch.Tensor, y: torch.Tensor, batch_size: int, rng: np.random.Generator) -> float:
    perm = torch.from_numpy(rng.permutation(X.shape[0]))
    correct = 0
    for s in range(0, X.shape[0], batch_size):
        idx = perm[s:s + batch_size]
        logits = head.fit_batch(X[idx], y[idx])
        correct += (logits.argmax(dim=1) == y[idx]).sum().item()
    return correct / max(1, X.shape[0])


def main():
    args = parse_args()
    if args.write_random_checkpoint:
        write_random_checkpoint(args.write_random_checkpoint, args.seed)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    head_device = torch.device(args.head_device)

    max_instances_description = str(args.max_instances) if args.max_instances is not None else "500"
    experiment_name = f"head_{args.method}_{args.experiment_type}_{args.fine_classes}_{max_instances_description}"
    if args.run_path is None:
        run_path = get_run_path(prefix=f"cifar_100/{experiment_name}", path=args.runs_root)
    else:
        run_path = args.run_path
    create_run_path(run_path)
    print(f"Experiment name: {experiment_name}")
    print(f"Run path: {run_path}")
    with open(os.path.join(run_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # ---- data (encodings) -----------------------------------------------------------
    enc = get_encodings(args, device)

    def to_t(X, y):
        return torch.from_numpy(X).float().to(head_device), torch.from_numpy(y).long().to(head_device)

    train_sets = {}
    for name in GROUPS:
        X, yc, yf = subsample_per_coarse_class(*enc[f"train_{name}"], args.coarse_classes, args.max_instances, rng)
        train_sets[name] = to_t(X, yc)
    test_sets = {name: to_t(enc[f"test_{name}"][0], enc[f"test_{name}"][1]) for name in GROUPS}
    for name in GROUPS:
        logger.info(f"train {name}: {train_sets[name][0].shape[0]}  test {name}: {test_sets[name][0].shape[0]}")

    # Epoch count exactly as the reference LTM script (independent of batch size)
    instances_per_epoch = 500 if args.max_instances is None else args.max_instances
    num_epochs = args.epochs if args.epochs is not None else int(TARGET_STEPS / instances_per_epoch)
    logger.info(f"Exp.:{args.experiment_type} Method:{args.method} Fine classes:{args.fine_classes} "
                f"Batch size:{args.batch_size} max. instances:{args.max_instances} LR:{args.lr} "
                f"epochs/phase:{num_epochs} pretrain epochs:{args.pretrain_epochs} seed:{args.seed}")

    # ---- head ----------------------------------------------------------------------
    if args.method == "sdmlp" and args.gaba_switch_activations is None:
        n_pretrain_samples = args.pretrain_epochs * train_sets["12"][0].shape[0]
        args.gaba_switch_activations = max(1, n_pretrain_samples // 2)
        logger.info(f"SDMLP GABA switch activations (auto): {args.gaba_switch_activations}")
    head = make_head(args, head_device)

    results_file = CifarResults(run_path=run_path, suffix=args.experiment_type)
    results_file.clear_file()
    results_pretrain = CifarResults(run_path=run_path, suffix="pretrain")
    results_pretrain.clear_file()
    timings = []

    def do_epoch(results, train_fine_classes: list[int], X, y, epoch: int, epoch_global: int, total_epochs: int):
        t = EpochTiming()
        t0 = time.time()
        train_acc = train_epoch(head, X, y, args.batch_size, rng)
        t.train_s = time.time() - t0
        print(f"Epoch {epoch + 1:3d}/{total_epochs} | train acc {train_acc:.3f} | {head.describe()}")
        results.append_line(
            coarse_classes=args.coarse_classes,
            fine_classes=train_fine_classes,
            mode=Instrumentation.MODE_TRAINING,
            epoch=epoch,
            accuracy=train_acc,
        )
        t0 = time.time()
        accs = {}
        for name in EVALUATE_NAMES:
            acc = evaluate(head, *test_sets[name])
            accs[name] = acc
            results.append_line(
                coarse_classes=args.coarse_classes,
                fine_classes=[name],
                mode=Instrumentation.MODE_EVALUATE,
                epoch=epoch_global,
                accuracy=acc,
            )
        t.eval_s = time.time() - t0
        print(f"Epoch {epoch + 1:3d}/{total_epochs} | " +
              " | ".join(f"eval. acc. {n}: {a:.3f}" for n, a in accs.items()) +
              f" | train {t.train_s:.2f}s eval {t.eval_s:.2f}s")
        timings.append({"phase": train_fine_classes, "epoch": epoch, "train_s": t.train_s, "eval_s": t.eval_s})
        return accs

    # ---- pretraining on fine-classes 1,2 (plays the role of the STM pretraining) ----
    if args.pretrain_epochs > 0:
        logger.info(f"Pretraining head on fine-classes 1,2 for {args.pretrain_epochs} epochs")
        X, y = train_sets["12"]
        for epoch in range(args.pretrain_epochs):
            do_epoch(results_pretrain, [1, 2], X, y, epoch, epoch, args.pretrain_epochs)

    # ---- continual phases -----------------------------------------------------------
    epoch_global = 0
    for fine_class in args.fine_classes:
        logger.info(f"Training fine class {fine_class}")
        X, y = train_sets[str(fine_class)]
        for epoch in range(num_epochs):
            do_epoch(results_file, [fine_class], X, y, epoch, epoch_global, num_epochs)
            epoch_global += 1

    with open(os.path.join(run_path, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    if timings:
        tr = np.mean([t["train_s"] for t in timings])
        ev = np.mean([t["eval_s"] for t in timings])
        print(f"Mean per-epoch wall-clock: train {tr:.3f}s, eval {ev:.3f}s (n={len(timings)} epochs)")
    print(f"Results written to {results_file.get_file_name()}")


if __name__ == "__main__":
    main()
