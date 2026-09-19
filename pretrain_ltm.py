"""
Cifar LTM pre-training (script version of cifar_ltm_pretraining.ipynb).

Pre-trains the ResNet-18 LTM (CifarClassifier) on fine sub-classes 1 and 2 of ALL coarse
classes, using the coarse label (20 classes) as the target, exactly as the notebook does:

    model:      CifarClassifier(ResNetConfig(num_classes=20), bias_stage=-1)
    optimizer:  Adam, lr = 1e-3
    batch size: 128
    epochs:     40 (notebook default; override with --epochs)
    train set:  train split, fine classes 1,2 (exclude Cifar100Dataset.get_fine_classes([3,4,5]))

Differences from the notebook (all additive):
  * After EVERY epoch the state dict is saved to
        <output-dir>/cifar_100_subclasses_12_e{epoch}.pth   (epoch is 1-based, e11 = after 11 epochs)
    so a specific epoch can be selected afterwards. The notebook only saved the final epoch
    as cifar_100_subclasses_12_e{num_epochs}.pth; the released checkpoint
    "cifar_100_subclasses_12_e11_31.1.pth" is an 11-epoch model whose accuracy on test fine
    classes 3,4,5 (all coarse) was 31.1% (see cifar_ltm_evaluate.ipynb).
  * Each epoch is evaluated on BOTH
        (a) test fine classes 1,2 (all coarse)  -> "acc_12"   (what the notebook printed)
        (b) test fine classes 3,4,5 (all coarse) -> "acc_345"  (what the evaluate notebook measured)
    and a table (epoch, train_acc, acc_12, acc_345) is printed and written to
        <output-dir>/cifar_100_subclasses_12_epochs.csv
  * Checkpoints are written with CPU tensors so they load on any device.
  * --device / --seed / --num-workers CLI args.

Run from the repo root, e.g.
    python pretrain_ltm.py --epochs 40
"""
import argparse
import csv
import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from environment.cifar.cifar_classifier import CifarClassifier
from environment.cifar.cifar_dataset import Cifar100Dataset
from model.resnet import ResNetConfig
from util.device import get_device
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    mean_loss: float = 0
    mean_accuracy: float = 0
    num_samples: int = 0
    global_step: int = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs (notebook: 40)")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size (notebook: 128)")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (notebook: 1e-3)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. cpu, mps, cuda. Default: util.device.get_device() (cuda > mps > cpu)")
    parser.add_argument("--data-path", type=str, default="../cifar-100-python",
                        help="Directory containing the CIFAR-100 python pickle files (train, test, meta)")
    parser.add_argument("--output-dir", type=str, default="../cifar_100_pretrain",
                        help="Where per-epoch checkpoints and the epoch table are written")
    parser.add_argument("--model-name", type=str, default="cifar_100_subclasses_12")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (notebook: 2)")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch seed (notebook: unseeded)")
    return parser.parse_args()


def do_epoch_mode(
    model,
    loader,
    optimizer,
    device,
    writer,
    training: bool,
    global_step: int,
    max_steps: int = 0,
    log_period: int = 100,
) -> EpochMetrics:
    """Identical to the notebook's do_epoch_mode, with the writer passed explicitly."""
    if training:
        model.train()
        epoch_type = "Training"
    else:
        model.eval()
        epoch_type = "Evaluate"

    epoch_loss = 0.0
    epoch_correct = 0
    epoch_samples = 0

    log_loss = 0.0
    log_correct = 0
    log_samples = 0
    num_steps = 0

    for x, y in loader:

        if max_steps > 0 and num_steps >= max_steps:
            break  # early truncation of epoch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad()

            logits, encoding = model(x, bias=None)

            loss = F.cross_entropy(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits, encoding = model(x, bias=None)

                loss = F.cross_entropy(
                    logits,
                    y,
                )

        samples_step = x.size(0)
        loss_step = (
            loss.item() * samples_step
        )
        epoch_loss += loss_step

        correct_step = (
            (logits.argmax(dim=1) == y)
            .sum()
            .item()
        )
        epoch_correct += correct_step
        epoch_samples += samples_step

        log_loss += loss_step
        log_correct += correct_step
        log_samples += samples_step

        if (global_step < 100) or ((log_samples % log_period) == 0):
            log_accuracy = log_correct / log_samples
            writer.add_scalar(f'{epoch_type} Loss {log_period}', loss_step, global_step)
            writer.add_scalar(f'{epoch_type} Accuracy {log_period}', log_accuracy, global_step)
            log_loss = 0.0
            log_correct = 0
            log_samples = 0

        global_step += 1
        num_steps += 1

    epoch_metrics = EpochMetrics(
        mean_loss=epoch_loss / epoch_samples,
        mean_accuracy=epoch_correct / epoch_samples,
        num_samples=epoch_samples,
        global_step=global_step,
    )
    return epoch_metrics


def main():
    args = parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    model_name = args.model_name
    data_file_path = args.data_path
    model_dir = args.output_dir

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    logger.info(f"Device: {device}")

    # DataLoader settings as in the notebook. pin_memory only does anything for CUDA.
    num_workers = args.num_workers
    pin_memory = device.type == "cuda"

    run_root_path = "cifar_100_pretrain"
    run_path = get_run_path(
        prefix=run_root_path,
        path="./runs",
    )
    create_run_path(run_path)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Tensorboard run path: {run_path}")
    logger.info(f"Checkpoint dir: {model_dir}")

    # Train on fine sub-classes 1, 2 (all coarse classes), coarse labels.
    training_exclude_classes_fine = Cifar100Dataset.get_fine_classes([3, 4, 5])
    # (a) Eval on fine sub-classes 1, 2 (notebook's evaluate set)
    evaluate_exclude_classes_fine_12 = training_exclude_classes_fine
    # (b) Eval on fine sub-classes 3, 4, 5 (the transfer set the paper reports 31.1% on)
    evaluate_exclude_classes_fine_345 = Cifar100Dataset.get_fine_classes([1, 2])

    dataset_training = Cifar100Dataset(
        file_path=data_file_path,
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=None,
        exclude_classes_fine=training_exclude_classes_fine,
        as_tensor=True,
    )
    dataset_evaluate_12 = Cifar100Dataset(
        file_path=data_file_path,
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=None,
        exclude_classes_fine=evaluate_exclude_classes_fine_12,
        as_tensor=True,
    )
    dataset_evaluate_345 = Cifar100Dataset(
        file_path=data_file_path,
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=None,
        exclude_classes_fine=evaluate_exclude_classes_fine_345,
        as_tensor=True,
    )
    logger.info(
        f"Dataset sizes: train(1,2)={len(dataset_training)} "
        f"test(1,2)={len(dataset_evaluate_12)} test(3,4,5)={len(dataset_evaluate_345)}"
    )

    loader_training = DataLoader(
        dataset_training,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loader_evaluate_12 = DataLoader(
        dataset_evaluate_12,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    loader_evaluate_345 = DataLoader(
        dataset_evaluate_345,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    config = ResNetConfig(
        num_classes=dataset_training.get_num_classes(),
    )
    model = CifarClassifier(config, bias_stage=-1).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"# parameters:{num_parameters}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=run_path)

    table_path = os.path.join(model_dir, f"{model_name}_epochs.csv")
    with open(table_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "loss_12", "acc_12", "loss_345", "acc_345", "checkpoint", "seconds"])

    rows = []
    global_step_training = 0
    global_step_evaluate = 0

    for epoch in range(num_epochs):
        t0 = time.time()

        metrics_training = do_epoch_mode(
            model, loader_training, optimizer, device, writer,
            training=True, global_step=global_step_training, max_steps=0,
        )
        global_step_training = metrics_training.global_step
        t_train = time.time() - t0

        metrics_evaluate_12 = do_epoch_mode(
            model, loader_evaluate_12, optimizer, device, writer,
            training=False, global_step=global_step_evaluate, max_steps=0,
        )
        global_step_evaluate = metrics_evaluate_12.global_step

        metrics_evaluate_345 = do_epoch_mode(
            model, loader_evaluate_345, optimizer, device, writer,
            training=False, global_step=global_step_evaluate, max_steps=0,
        )

        writer.add_scalar('Training-Loss', metrics_training.mean_loss, global_step_training)
        writer.add_scalar('Training-Accuracy', metrics_training.mean_accuracy, global_step_training)
        writer.add_scalar('Evaluate-Loss', metrics_evaluate_12.mean_loss, global_step_evaluate)
        writer.add_scalar('Evaluate-Accuracy', metrics_evaluate_12.mean_accuracy, global_step_evaluate)
        writer.add_scalar('Evaluate-Loss-345', metrics_evaluate_345.mean_loss, global_step_evaluate)
        writer.add_scalar('Evaluate-Accuracy-345', metrics_evaluate_345.mean_accuracy, global_step_evaluate)

        # Save a checkpoint after every epoch (1-based epoch number in the file name).
        epoch_number = epoch + 1
        checkpoint_path = os.path.join(model_dir, f"{model_name}_e{epoch_number}.pth")
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, checkpoint_path)

        seconds = time.time() - t0
        print(
            f"Epoch {epoch_number:3d}/{num_epochs} "
            f"| train loss {metrics_training.mean_loss:.4f} "
            f"| train acc {metrics_training.mean_accuracy:.3f} "
            f"| test(1,2) loss {metrics_evaluate_12.mean_loss:.4f} "
            f"| test(1,2) acc {metrics_evaluate_12.mean_accuracy:.3f} "
            f"| test(3,4,5) loss {metrics_evaluate_345.mean_loss:.4f} "
            f"| test(3,4,5) acc {metrics_evaluate_345.mean_accuracy:.3f} "
            f"| {seconds:.1f}s (train {t_train:.1f}s) -> {checkpoint_path}",
            flush=True,
        )

        row = [
            epoch_number,
            f"{metrics_training.mean_loss:.4f}", f"{metrics_training.mean_accuracy:.4f}",
            f"{metrics_evaluate_12.mean_loss:.4f}", f"{metrics_evaluate_12.mean_accuracy:.4f}",
            f"{metrics_evaluate_345.mean_loss:.4f}", f"{metrics_evaluate_345.mean_accuracy:.4f}",
            checkpoint_path, f"{seconds:.1f}",
        ]
        rows.append(row)
        with open(table_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    writer.close()

    # Summary table
    print("\nepoch  train_acc  acc_12  acc_345  checkpoint")
    for row in rows:
        print(f"{int(row[0]):5d}  {float(row[2]):9.4f}  {float(row[4]):6.4f}  {float(row[6]):7.4f}  {row[7]}")
    best_12 = max(rows, key=lambda r: float(r[4]))
    print(f"\nBest acc_12 (test fine classes 1,2): epoch {best_12[0]} acc_12={best_12[4]} acc_345={best_12[6]} -> {best_12[7]}")
    print(f"Epoch table written to: {table_path}")
    print("Models saved successfully.")


if __name__ == "__main__":
    main()
