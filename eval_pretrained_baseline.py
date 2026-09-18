"""
Evaluate the pre-trained (not yet continually fine-tuned) LTM on the four continual-learning test sets
and write the result as a CifarResults file. This gives the "before the continual phase" accuracies
R[0, j] (the baseline b_j used by the FWT / BWT_0 metrics in plot_comparison.py) for the LTM-only method,
whose script (cifar_main_ltm_fine_tuning.py) only evaluates after each training epoch.

Evaluation is done exactly as cifar_main_ltm_fine_tuning.py does it: CifarClassifier(ResNetConfig(num_classes=20),
bias_stage=-1), state_dict from --checkpoint, model.eval(), coarse-label 20-way argmax on the test images of the
given coarse-class pair restricted to fine-class groups "12" (fine classes 1,2), "3", "4" and "5". It runs on CPU
(no MPS / CUDA) so it can be used while a GPU pipeline is running.

Output (default runs_local/baselines/ltm_pretrained_baseline.txt), one line per test set:

    [0, 1], ['12'], evaluate, 0, <accuracy>

i.e. mode "evaluate", epoch 0, which CifarResults.read_results_file reads as Epoch 1.

Example:
    python eval_pretrained_baseline.py --checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth
"""
import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from environment.cifar.cifar_classifier import CifarClassifier
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_results import CifarResults
from model.resnet import ResNetConfig
from util.instrumentation import Instrumentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 20
EVALUATE_NAMES = ["12", "3", "4", "5"]
GROUPS = {"12": [1, 2], "3": [3], "4": [4], "5": [5]}
ALL_GROUPS = [1, 2, 3, 4, 5]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default="../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth",
                   help="LTM state_dict (CifarClassifier(ResNetConfig(num_classes=20), bias_stage=-1)).")
    p.add_argument("--data-path", type=str, default="../cifar-100-python")
    p.add_argument("--coarse-classes", type=int, nargs="+", default=[0, 1])
    p.add_argument("--batch-size", type=int, default=16, help="As cifar_main_ltm_fine_tuning.py (does not affect the result).")
    p.add_argument("--out", type=str, default="runs_local/baselines/ltm_pretrained_baseline.txt",
                   help="CifarResults-format output file (its directory is created).")
    p.add_argument("--threads", type=int, default=2, help="torch CPU threads (kept small to stay out of the way).")
    return p.parse_args()


def build_test_dataset(data_path: str, coarse_classes: list[int], group: list[int]) -> Cifar100Dataset:
    """Same construction as dataset_evaluate_* in cifar_main_ltm_fine_tuning.py (all test instances)."""
    exclude_fine = Cifar100Dataset.get_fine_classes([g for g in ALL_GROUPS if g not in group])
    exclude_coarse = Cifar100Dataset.get_coarse_classes_excluded(coarse_classes)
    return Cifar100Dataset(
        file_path=data_path,
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=exclude_coarse,
        exclude_classes_fine=exclude_fine,
        as_tensor=True,
    )


def release_shared_memory(dataset: Cifar100Dataset):
    """Cifar100Dataset keeps its arrays in multiprocessing shared memory and never frees them."""
    for shm in (dataset.shared_memory_images, dataset.shared_memory_labels_coarse, dataset.shared_memory_labels_fine):
        if shm is None:
            continue
        try:
            shm.close()
            shm.unlink()
        except Exception:  # noqa: BLE001 - best effort
            pass


@torch.no_grad()
def evaluate(model: CifarClassifier, loader: DataLoader, device: torch.device) -> float:
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, bias=None)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return correct / max(1, total)


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")  # deliberately CPU only, see module docstring

    model = CifarClassifier(ResNetConfig(num_classes=NUM_CLASSES), bias_stage=-1)
    logger.info(f"Loading LTM from {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    for name in EVALUATE_NAMES:
        ds = build_test_dataset(args.data_path, args.coarse_classes, GROUPS[name])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        acc = evaluate(model, loader, device)
        release_shared_memory(ds)
        logger.info(f"test set {name}: {len(ds)} instances, accuracy {acc:.4f}")
        lines.append(CifarResults.get_line(
            coarse_classes=args.coarse_classes,
            fine_classes=[name],
            mode=Instrumentation.MODE_EVALUATE,
            epoch=0,
            accuracy=acc,
        ))
    with open(args.out, "w") as f:
        f.writelines(lines)
    logger.info(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
