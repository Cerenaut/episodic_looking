"""
CLS/STM trained by supervised learning, without episodes ("sup"; next_draft_experiments.tex, Models).

The same STM network as the RL model (cifar_main_stm_training.py: SparseDistributedModel, LayerNorm input, top-32 of
1,000 hidden cells, 3 layers, leaky ReLU; bias = tanh of the clamped output, gate 2*sigmoid(bias) on the 128 channels
of the frozen LTM's stage 2), trained by cross-entropy back-propagated through the frozen LTM. Without RL there is no
episode, critic, policy sampling or reward. Per image (or minibatch):

  1. look      the frozen LTM, no bias, no gradient: stage-2 encoding e0 (128) and logits z0 (20)
  2. bias      b = STM([e0, z0])  (the RL STM's input at the first step of an episode, without its always-zero bias
               slot and empty history)
  3. perceive  the frozen LTM with gate 2*sigmoid(b), with gradient to b
  4. learn     cross-entropy on those logits; one optimizer step per minibatch (SGD, momentum 0.5, as the RL STM)

The STM's output layer starts at zero, so before training the gate is exactly 1 and the model is exactly the LTM.
Step 1 depends only on the image (the LTM is frozen, in eval mode, and there is no augmentation), so it is computed
once per dataset, at batch 256, and stored beside the images.

Epochs are passes over the training images, as for the heads and LTM-only (there are no episodes to count). The data,
validation split, seeding, evaluation sets and results files follow cifar_main_ltm_fine_tuning.py, and the experiment
types and checkpoints follow the STM script, so run_v2.sh and metrics_v2.py treat it as the other CLS/STM models:

  pretrain   fine 1,2; saves --stm-checkpoint (created from --seed)
  evaluate   the four test sets (and validation sets) before the continual phases (baseline unit)
  continual  --fine-classes in order, from --stm-checkpoint; --stm-checkpoint-out saves after each phase
"""
import logging
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from environment.cifar.cifar_args import CifarArgs
from environment.cifar.cifar_classifier import CifarClassifier
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_results import CifarResults
from model.resnet import ResNetConfig
from model.sparse.sparse_dense_model import SparseActivationDenseModelConfig
from model.sparse.sparse_distributed_model import SparseDistributedModel
from util.device import get_device, seed_all
from util.instrumentation import Instrumentation
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 20
ENCODING_SIZE = 128  # stage-2 channels, pooled
BIAS_SIZE = 128
BIAS_STAGE = 2
MAX_LOGIT_MAGNITUDE = 10.0  # PolicyConfig.max_logit_magnitude, as the RL STM's policy mean
MOMENTUM = 0.5  # the RL STM's optimizer (CifarAgentConfig momentum in cifar_main_stm_training.py)
EVALUATE_NAMES = ["12", "3", "4", "5"]
EVALUATE_BATCH_SIZE = 256  # evaluation only (no training effect: the LTM is in eval mode and the STM has no BN)


class SupervisedSTM(nn.Module):
    """The RL STM's actor network (CifarModel.create_model, sparse branch), reading [e0, z0], emitting the bias."""

    def __init__(self, sparsity: int, hidden_size: int = 1000, layers: int = 3):
        super().__init__()
        input_size = ENCODING_SIZE + NUM_CLASSES
        model_config = SparseActivationDenseModelConfig(
            weight_decay_factor = 0,
            name = "Actor",
            nonlinearity = "leaky-relu",
            input_layer_norm = True,
            input_dropout = 0,
            input_weight_clip = 0.0,
            input_size = input_size,
            hidden_size = hidden_size,
            hidden_size_factor = 1,
            output_size = BIAS_SIZE,
            output_nonlinearity = None,
            layers = layers,
            hidden_dropout = 0,
            bias = True,
        )
        self.model = SparseDistributedModel(
            model_configs = [model_config],
            input_key_size = input_size,
            input_value_size = input_size,
            memory_size = hidden_size,
            ensemble_size = 1,
            sparsity = sparsity,
        )
        output_layer = self.model.get_model(0).layers[-1]
        with torch.no_grad():  # gate exactly 1 before training: the untrained model is the LTM
            output_layer.weight.zero_()
            output_layer.bias.zero_()

    def get_trainable_parameters(self) -> list:
        return self.model.get_trainable_parameters()

    def forward(self, encoding: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        x = torch.cat([encoding, logits], dim=1)
        output, _ = self.model.do_model(key_input = x, model_input = x)
        return torch.tanh(torch.clamp(output, -MAX_LOGIT_MAGNITUDE, MAX_LOGIT_MAGNITUDE))


def main():
    cifar_args = CifarArgs.parse_args()

    SEED = cifar_args.seed
    if SEED is not None:
        seed_all(SEED)  # the STM's initial weights and fixed projection, and anything else not seeded below

    EXPERIMENT_TYPE = cifar_args.experiment_type
    BATCH_SIZE = cifar_args.batch_size
    MAX_INSTANCES = cifar_args.max_instances
    FINE_CLASSES = cifar_args.fine_classes
    COARSE_CLASSES = cifar_args.coarse_classes
    LEARNING_RATE = cifar_args.learning_rate
    VAL_HOLDOUT = cifar_args.val_holdout
    LOADER_WORKERS = cifar_args.loader_workers
    EVALUATE_INTERVAL_EPOCHS = cifar_args.evaluate_epochs
    PRETRAIN = EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_PRETRAIN
    EVALUATE_ONLY = EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_EVALUATE

    logger.info(f"Exp.:{EXPERIMENT_TYPE} Fine classes:{FINE_CLASSES} Batch size:{BATCH_SIZE} max. instances:{MAX_INSTANCES} "
                f"LR: {LEARNING_RATE} Seed: {SEED}")

    CIFAR_DATA_FILE_PATH = "../cifar-100-python"
    CLASSIFIER_FILE_PATH = "../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth"
    if cifar_args.ltm_checkpoint is not None:
        CLASSIFIER_FILE_PATH = cifar_args.ltm_checkpoint
    STM_CHECKPOINT = cifar_args.stm_checkpoint
    if STM_CHECKPOINT is None:
        raise ValueError("--stm-checkpoint is required (written by pretrain, read by the other types)")

    max_instances_description = str(MAX_INSTANCES) if MAX_INSTANCES is not None else "500"
    EXPERIMENT_NAME = f"{EXPERIMENT_TYPE}_{FINE_CLASSES}_{max_instances_description}"
    run_path = get_run_path(prefix = f"cifar_100/{EXPERIMENT_NAME}", path = cifar_args.run_root)
    create_run_path(run_path)
    data_file_path = CIFAR_DATA_FILE_PATH

    results_file = CifarResults(run_path = run_path, suffix = EXPERIMENT_TYPE)
    results_file.clear_file()
    results_file_val = None
    if VAL_HOLDOUT > 0:
        results_file_val = CifarResults(run_path = run_path, suffix = f"{EXPERIMENT_TYPE}_val")
        results_file_val.clear_file()

    device = get_device()
    logger.info(f"Device: {device}")

    if MAX_INSTANCES is None:
        instances_per_epoch = 500
    else:
        instances_per_epoch = MAX_INSTANCES
    NUM_EPOCHS = int(6250 / instances_per_epoch)  # 12 at 500 instances, as the heads and LTM-only
    if cifar_args.epochs is not None:
        NUM_EPOCHS = cifar_args.epochs
    logger.info(f"Epochs per phase: {NUM_EPOCHS}")
    EVALUATE_EPOCHS_SET = None
    if cifar_args.evaluate_points is not None:
        EVALUATE_EPOCHS_SET = CifarResults.evaluation_epochs(NUM_EPOCHS, cifar_args.evaluate_points)

    exclude_classes_coarse = {i for i in range(NUM_CLASSES) if i not in COARSE_CLASSES}
    exclude_fine = {
        "12": Cifar100Dataset.get_fine_classes([    3,4,5,]),
        "3":  Cifar100Dataset.get_fine_classes([1,2,  4,5,]),
        "4":  Cifar100Dataset.get_fine_classes([1,2,3,  5,]),
        "5":  Cifar100Dataset.get_fine_classes([1,2,3,4,  ]),
    }
    split_options = Cifar100Dataset.get_split_options(VAL_HOLDOUT, cifar_args.split_seed)
    subset_options = Cifar100Dataset.get_subset_options(MAX_INSTANCES, SEED)

    # Frozen LTM: eval mode (BN statistics fixed), no parameter gradients; gradients reach the bias through it
    ltm = CifarClassifier(ResNetConfig(num_classes = NUM_CLASSES), bias_stage = BIAS_STAGE).to(device)
    ltm.load_state_dict(torch.load(CLASSIFIER_FILE_PATH, weights_only = True))
    ltm.eval()
    for p in ltm.parameters():
        p.requires_grad_(False)

    def look(dataset) -> TensorDataset:
        """Step 1 for every image of a dataset: (image, e0, z0, label), in the dataset's order."""
        xs, es, zs, ys = [], [], [], []
        with torch.no_grad():
            for x, y in DataLoader(dataset, batch_size = 256, shuffle = False, num_workers = LOADER_WORKERS):
                x = x.to(device)
                logits_0, encoding_0 = ltm(x, bias = torch.zeros(x.shape[0], BIAS_SIZE, device = device))
                xs.append(x.cpu()); es.append(encoding_0.cpu()); zs.append(logits_0.cpu()); ys.append(y.cpu())
        return TensorDataset(torch.cat(xs), torch.cat(es), torch.cat(zs), torch.cat(ys))

    def sampler_generator(key: int):
        return None if SEED is None else torch.Generator().manual_seed(SEED * 10 + key)

    def training_loader(name: str, key: int, max_instances):
        dataset = Cifar100Dataset(
            file_path = data_file_path,
            label_type = Cifar100Dataset.LABEL_TYPE_COARSE,
            training = True,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_fine[name],
            max_instances = max_instances,
            as_tensor = True,
            **split_options,
            **(subset_options if max_instances is not None else {}),
        )
        return DataLoader(look(dataset), batch_size = BATCH_SIZE, shuffle = True, num_workers = LOADER_WORKERS,
                          pin_memory = True, generator = sampler_generator(key))

    def evaluation_loader(name: str, validation: bool):
        dataset = Cifar100Dataset(
            file_path = data_file_path,
            label_type = Cifar100Dataset.LABEL_TYPE_COARSE,
            training = validation,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_fine[name],
            as_tensor = True,
            **({"split_part": "validation", **split_options} if validation else {}),
        )
        return DataLoader(look(dataset), batch_size = EVALUATE_BATCH_SIZE, shuffle = False, num_workers = LOADER_WORKERS,
                          pin_memory = True)

    # Pre-training: fine 1,2 of the pair, all images (MAX_INSTANCES applies to the later phases), the '1,2' sets only.
    # Otherwise: each trained class, and the four sets.
    evaluate_names = ["12"] if PRETRAIN else EVALUATE_NAMES
    loaders_evaluate = {name: evaluation_loader(name, validation = False) for name in evaluate_names}
    loaders_validation = {}
    if VAL_HOLDOUT > 0:
        loaders_validation = {name: evaluation_loader(name, validation = True) for name in evaluate_names}
    loaders_training = {}
    if PRETRAIN:
        loaders_training[12] = training_loader("12", 12, None)
    elif not EVALUATE_ONLY:
        for c in FINE_CLASSES:
            loaders_training[c] = training_loader(str(c), c, MAX_INSTANCES)
    logger.info(f"Training set sizes: {[(k, len(l.dataset)) for k, l in loaders_training.items()]}; validation set "
                f"sizes: {[(k, len(l.dataset)) for k, l in loaders_validation.items()]}")

    stm = SupervisedSTM(sparsity = cifar_args.sparsity).to(device)
    if not PRETRAIN:
        logger.info(f"Loading STM from {STM_CHECKPOINT}")
        stm.load_state_dict(torch.load(STM_CHECKPOINT, weights_only = True))
    optimizer = torch.optim.SGD(stm.get_trainable_parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    logger.info(f"STM trainable parameters: {sum(p.numel() for p in stm.get_trainable_parameters())}")

    writer = SummaryWriter(log_dir = run_path)

    def perceive(x: torch.Tensor, encoding_0: torch.Tensor, logits_0: torch.Tensor) -> torch.Tensor:
        """Steps 2 and 3: logits of the LTM gated by the STM's bias (the graph to the STM is kept when grad is enabled)."""
        bias = stm(encoding_0, logits_0)
        logits, _ = ltm(x, bias = bias)
        return logits

    @dataclass
    class EpochMetrics:
        mean_loss: float
        mean_accuracy: float

    def do_epoch_mode(loader, training: bool) -> EpochMetrics:
        stm.train(training)  # no dropout or BN in the STM: this only documents the mode
        total_loss = torch.zeros((), device = device)  # summed on the device: no host sync per minibatch
        correct = torch.zeros((), device = device, dtype = torch.long)
        samples = 0
        for x, encoding_0, logits_0, y in loader:
            x, encoding_0, logits_0, y = (t.to(device, non_blocking = True) for t in (x, encoding_0, logits_0, y))
            if training:
                optimizer.zero_grad()
                logits = perceive(x, encoding_0, logits_0)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits = perceive(x, encoding_0, logits_0)
                    loss = F.cross_entropy(logits, y)
            total_loss += loss.detach() * x.size(0)
            correct += (logits.argmax(dim = 1) == y).sum()
            samples += x.size(0)
        return EpochMetrics(total_loss.item() / samples, correct.item() / samples)

    def evaluate(epoch: int, step: int):
        for name in evaluate_names:
            m = do_epoch_mode(loaders_evaluate[name], training = False)
            writer.add_scalar(f"Eval.-Acc.-{name}", m.mean_accuracy, step)
            print(f"Epoch {epoch + 1:3d} | eval. acc. {name}: {m.mean_accuracy:.3f}")
            # Pre-training rows name the classes as the STM script does ([1, 2]); otherwise the set names
            fine = [1, 2] if PRETRAIN else [name]
            results_file.append_line(COARSE_CLASSES, fine, Instrumentation.MODE_EVALUATE, epoch, m.mean_accuracy)
            if name in loaders_validation:
                v = do_epoch_mode(loaders_validation[name], training = False)
                print(f"Epoch {epoch + 1:3d} | val. acc. {name}: {v.mean_accuracy:.3f}")
                results_file_val.append_line(COARSE_CLASSES, fine, Instrumentation.MODE_EVALUATE, epoch, v.mean_accuracy)

    if EVALUATE_ONLY:
        evaluate(epoch = 0, step = 0)
        return

    def save(path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
        torch.save(stm.state_dict(), path)
        logger.info(f"Saved STM to {path}")

    phases = [(12, [1, 2])] if PRETRAIN else [(c, [c]) for c in FINE_CLASSES]
    step = 0
    for key, fine in phases:
        logger.info(f"Training fine class(es) {fine}")
        for epoch in range(NUM_EPOCHS):
            m = do_epoch_mode(loaders_training[key], training = True)
            step += 1
            writer.add_scalar("Training-Loss", m.mean_loss, step)
            writer.add_scalar("Training-Accuracy", m.mean_accuracy, step)
            print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | train loss {m.mean_loss:.4f} | train acc {m.mean_accuracy:.3f}")
            for results in (results_file, results_file_val):
                if results is not None:
                    results.append_line(COARSE_CLASSES, fine, Instrumentation.MODE_TRAINING, epoch, m.mean_accuracy)
            if EVALUATE_EPOCHS_SET is not None:
                evaluate_now = epoch in EVALUATE_EPOCHS_SET
            else:
                evaluate_now = EVALUATE_INTERVAL_EPOCHS <= 1 or epoch % EVALUATE_INTERVAL_EPOCHS == 0 or epoch == NUM_EPOCHS - 1
            if evaluate_now:
                evaluate(epoch, step)
        if not PRETRAIN and cifar_args.stm_checkpoint_out is not None:
            out = cifar_args.stm_checkpoint_out
            save(os.path.join(out, f"stm_after_fine{key}.pth") if len(phases) > 1 else out)
    if PRETRAIN:
        save(STM_CHECKPOINT)


if __name__ == "__main__":
    main()
