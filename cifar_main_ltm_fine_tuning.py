import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from environment.cifar.cifar_args import CifarArgs
from environment.cifar.cifar_classifier import CifarClassifier
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_experiment import LtmExperimentConfig
from environment.cifar.cifar_results import CifarResults
from model.resnet import ResNetConfig
from util.device import get_device, seed_all
from util.instrumentation import Instrumentation
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    cifar_args = CifarArgs.parse_ltm_args()
    experiment = LtmExperimentConfig.from_args(cifar_args)

    SEED = experiment.seed
    if SEED is not None:
        # Model state is loaded from the checkpoint; the randomness is the training loaders' shuffles (seeded below
        # through their own generators) and any other torch / numpy use.
        seed_all(SEED)

    EXPERIMENT_TYPE = experiment.experiment_type
    BATCH_SIZE = experiment.batch_size
    MAX_INSTANCES = experiment.max_instances
    FINE_CLASSES = experiment.fine_classes
    COARSE_CLASSES = experiment.coarse_classes
    LEARNING_RATE = experiment.learning_rate
    VAL_HOLDOUT = experiment.val_holdout
    LOADER_WORKERS = experiment.loader_workers
    EVALUATE_INTERVAL_EPOCHS = experiment.evaluate_interval_epochs  # evaluate at epoch % k == 0 and the last epoch of a phase
    BN_MODE = experiment.bn_mode  # train (default, the draft's) | frozen (eval-mode BN, affine parameters fixed)

    logger.info(f"Exp.:{EXPERIMENT_TYPE} Fine classes:{FINE_CLASSES} Batch size:{BATCH_SIZE} max. instances:{MAX_INSTANCES} LR: {LEARNING_RATE} Seed: {SEED} BN mode: {BN_MODE}")

    CLASSIFIER_FILE_PATH = experiment.ltm_checkpoint
    NUM_CLASSES = experiment.NUM_CLASSES

    EXPERIMENT_NAME = experiment.experiment_name
    print(f"Experiment name: {EXPERIMENT_NAME}")

    run_root_path = f"cifar_100/{EXPERIMENT_NAME}"
    run_path = get_run_path(
        prefix = run_root_path, 
        path = experiment.run_root,  # honour --run-root as the STM and head scripts do (default ./runs)
    )
    create_run_path(run_path)
    data_file_path = experiment.CIFAR_DATA_FILE_PATH

    results_file = CifarResults(run_path = run_path, suffix=EXPERIMENT_TYPE)
    results_file.clear_file()
    results_file_val = None  # same format and epochs as results_file, on the validation sets
    if VAL_HOLDOUT > 0:
        results_file_val = CifarResults(run_path = run_path, suffix=f"{EXPERIMENT_TYPE}_val")
        results_file_val.clear_file()

    device = get_device()  # cuda > mps > cpu; override with EPISODIC_DEVICE env var
    print(f"Device:{device}")

    MAX_STEPS_TRAINING = 0  # measure in epochs
    MAX_STEPS_EVALUATE = 0  # whole epoch

    NUM_EPOCHS = experiment.num_epochs  # int(6250 / instances per epoch) unless --epochs (LtmExperimentConfig)
    EVALUATE_EPOCHS_SET = experiment.evaluate_epochs_set  # --evaluate-points: log-spaced evaluation epochs within each phase

    # Select coarse classes
    exclude_classes_coarse = set()
    for i in range(NUM_CLASSES):
        if i not in COARSE_CLASSES:
            exclude_classes_coarse.add(i)
    logger.info(f"Using coarse classes: {COARSE_CLASSES} excluding: {exclude_classes_coarse}")

    exclude_classes_fine_12 = Cifar100Dataset.get_fine_classes([    3,4,5,])
    exclude_classes_fine_3  = Cifar100Dataset.get_fine_classes([1,2,  4,5,])
    exclude_classes_fine_4  = Cifar100Dataset.get_fine_classes([1,2,3,  5,])
    exclude_classes_fine_5  = Cifar100Dataset.get_fine_classes([1,2,3,4,  ])

    # Validation split (--val-holdout): the training sets hold out the same images per fine class as the head script,
    # and the held-out images are evaluated alongside the test sets. The epoch count stays nominal (500 instances).
    # With --max-instances and --seed, the subsets are those the head and STM scripts draw for that seed.
    training_options = experiment.training_dataset_options
    validation_options = experiment.validation_dataset_options  # None: no split
    if validation_options is not None:
        logger.info(f"Validation split: {VAL_HOLDOUT} images per fine class held out, split seed {experiment.split_seed}")
    if training_options.subset_seed is not None:
        logger.info(f"Subsets of {MAX_INSTANCES} images per coarse class drawn with seed {SEED}")

    dataset_training_3 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_3,
        max_instances = MAX_INSTANCES,
        as_tensor=True,
        **training_options.get_constructor_options(),
    )
    dataset_training_4 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_4,
        max_instances = MAX_INSTANCES,
        as_tensor=True,
        **training_options.get_constructor_options(),
    )
    dataset_training_5 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_5,
        max_instances = MAX_INSTANCES,
        as_tensor=True,
        **training_options.get_constructor_options(),
    )

    # Evaluate on all instances in epoch
    dataset_evaluate_12 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_12,
        as_tensor=True,
    )
    dataset_evaluate_3 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_3,
        as_tensor=True,
    )
    dataset_evaluate_4 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_4,
        as_tensor=True,
    )
    dataset_evaluate_5 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=False,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_5,
        as_tensor=True,
    )

    pin_memory = True

    def sampler_generator(fine_class:int):
        # Each training loader shuffles with its own generator when seeded, so its order does not depend on other
        # torch random use. None = the global generator, as before.
        if SEED is None:
            return None
        return torch.Generator().manual_seed(SEED * 10 + fine_class)

    loader_training_3 = DataLoader(
        dataset_training_3,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
        generator=sampler_generator(3),
    )
    loader_training_4 = DataLoader(
        dataset_training_4,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
        generator=sampler_generator(4),
    )
    loader_training_5 = DataLoader(
        dataset_training_5,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
        generator=sampler_generator(5),
    )

    loader_evaluate_12 = DataLoader(
        dataset_evaluate_12,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
    )
    loader_evaluate_3 = DataLoader(
        dataset_evaluate_3,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
    )
    loader_evaluate_4 = DataLoader(
        dataset_evaluate_4,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
    )
    loader_evaluate_5 = DataLoader(
        dataset_evaluate_5,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=LOADER_WORKERS,
        pin_memory=pin_memory,
    )

    # Validation sets: the held-out training images of each test group
    loaders_validation = []
    if validation_options is not None:
        for exclude_classes_fine_validation in (exclude_classes_fine_12, exclude_classes_fine_3, exclude_classes_fine_4, exclude_classes_fine_5):
            dataset_validation = Cifar100Dataset(
                file_path=data_file_path,
                label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
                training=validation_options.training,
                exclude_classes_coarse=exclude_classes_coarse,
                exclude_classes_fine=exclude_classes_fine_validation,
                as_tensor=True,
                **validation_options.get_constructor_options(),
            )
            loaders_validation.append(DataLoader(
                dataset_validation,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=LOADER_WORKERS,
                pin_memory=pin_memory,
            ))
        logger.info(f"Training set sizes: 3: {len(dataset_training_3)} 4: {len(dataset_training_4)} 5: {len(dataset_training_5)}; "
                    f"validation set sizes: {[len(loader.dataset) for loader in loaders_validation]}")

    config = ResNetConfig(
        num_classes=NUM_CLASSES,
    )
    model = CifarClassifier(config, bias_stage=-1).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"# parameters:{num_parameters}")

    logger.info(f"Loading conv. model from file: {CLASSIFIER_FILE_PATH}")
    state_dict = torch.load(CLASSIFIER_FILE_PATH, weights_only=True)
    model.load_state_dict(state_dict)

    # --bn-mode frozen: every BatchNorm layer stays in eval mode while the rest of the model trains (running statistics
    # fixed, set_training_mode below) and its affine parameters are not trained (requires_grad off and left out of the
    # optimizer). train: unchanged, the optimizer gets model.parameters() exactly as before.
    bn_modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if BN_MODE == "frozen":
        for m in bn_modules:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"BN frozen: {len(bn_modules)} BatchNorm layers in eval mode, "
                    f"{sum(p.numel() for m in bn_modules for p in m.parameters(recurse=False))} affine parameters fixed; "
                    f"{sum(p.numel() for p in trainable)} parameters trained")
        optimizer = torch.optim.Adam(trainable, lr=LEARNING_RATE)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def set_training_mode(model):
        model.train()
        if BN_MODE == "frozen":
            for m in bn_modules:
                m.eval()
    writer = SummaryWriter(log_dir=run_path)

    @dataclass
    class EpochMetrics:
        mean_loss:float = 0
        mean_accuracy:float = 0
        num_samples:int = 0
        global_step:int = 0

    def do_epoch_mode(
        model,
        loader,
        optimizer,
        device,
        training:bool,
        global_step:int,
        max_steps:int = 0,
        log_period:int = 100,
    ) -> EpochMetrics:
        if training:
            set_training_mode(model)  # model.train(), then BN back to eval mode if --bn-mode frozen
        else:
            model.eval()
            
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        num_steps = 0

        # Every time we iter the loader we get a fresh shuffle, even if truncated early.
        for x, y in loader:

            # Forward pass on model
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad()

                logits, encoding = model(x, bias=None)

                loss = F.cross_entropy(
                    logits,
                    y,
                )

                # Backward pass on model
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    logits, encoding = model(x, bias=None)

                    loss = F.cross_entropy(
                        logits,
                        y,
                    )

            # Instrumentation
            samples_step = x.size(0)
            loss_step  = (
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

            global_step += 1
            num_steps += 1

        epoch_metrics = EpochMetrics(
            mean_loss = epoch_loss / epoch_samples,
            mean_accuracy =  epoch_correct / epoch_samples,
            num_samples = epoch_samples,
            global_step = global_step,
        )
        return epoch_metrics

    def do_epoch(
        epoch:int,
        global_step_training:int,
        global_step_evaluate:int,
        loader_training,
    ):
        metrics_training = do_epoch_mode(
            model,
            loader_training,
            optimizer,
            device,
            training = True,
            global_step = global_step_training,
            max_steps = MAX_STEPS_TRAINING,
        )
        global_step_training = metrics_training.global_step

        writer.add_scalar('Training-Loss', metrics_training.mean_loss, global_step_training)
        writer.add_scalar('Training-Accuracy', metrics_training.mean_accuracy, global_step_training)

        print(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} "
            f"| train loss {metrics_training.mean_loss:.4f} "
            f"| train acc {metrics_training.mean_accuracy:.3f} "
        )
        for results in (results_file, results_file_val):
            if results is None:
                continue
            results.append_line(
                coarse_classes = COARSE_CLASSES,
                fine_classes = FINE_CLASSES,
                mode = Instrumentation.MODE_TRAINING,
                epoch = epoch,
                accuracy = metrics_training.mean_accuracy,            
            )

        evaluate_names = ["12","3","4","5"]
        evaluate_loaders = [loader_evaluate_12, loader_evaluate_3, loader_evaluate_4, loader_evaluate_5]
        if EVALUATE_EPOCHS_SET is not None:
            evaluate_now = epoch in EVALUATE_EPOCHS_SET
        else:
            evaluate_now = EVALUATE_INTERVAL_EPOCHS <= 1 or epoch % EVALUATE_INTERVAL_EPOCHS == 0 or epoch == NUM_EPOCHS - 1
        if not evaluate_now:
            evaluate_loaders = []  # training line only; the metrics count epochs by training lines
        for i, evaluate_loader in enumerate(evaluate_loaders):
            metrics_evaluate = do_epoch_mode(
                model,
                evaluate_loader,
                optimizer,
                device,
                training = False,
                global_step = global_step_evaluate,
                max_steps = MAX_STEPS_EVALUATE,
            )
            if i == 0:    
                global_step_evaluate = metrics_evaluate.global_step

            evaluate_name = evaluate_names[i]
            writer.add_scalar(f'Eval.-Loss-{evaluate_name}', metrics_evaluate.mean_loss, global_step_evaluate)
            writer.add_scalar(f'Eval.-Acc.-{evaluate_name}', metrics_evaluate.mean_accuracy, global_step_evaluate)

            print(
                f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} "
                f"| eval. loss {evaluate_name}: {metrics_evaluate.mean_loss:.4f} "
                f"| eval. acc. {evaluate_name}: {metrics_evaluate.mean_accuracy:.3f}"
            )
            results_file.append_line(
                coarse_classes = COARSE_CLASSES,
                fine_classes = [evaluate_name],
                mode = Instrumentation.MODE_EVALUATE,
                epoch = epoch,
                accuracy = metrics_evaluate.mean_accuracy,            
            )

        for i, validation_loader in enumerate(loaders_validation if evaluate_loaders else []):
            metrics_validation = do_epoch_mode(
                model,
                validation_loader,
                optimizer,
                device,
                training = False,
                global_step = global_step_evaluate,
                max_steps = MAX_STEPS_EVALUATE,
            )
            print(
                f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} "
                f"| val. acc. {evaluate_names[i]}: {metrics_validation.mean_accuracy:.3f}"
            )
            results_file_val.append_line(
                coarse_classes = COARSE_CLASSES,
                fine_classes = [evaluate_names[i]],
                mode = Instrumentation.MODE_EVALUATE,
                epoch = epoch,
                accuracy = metrics_validation.mean_accuracy,
            )

        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} complete.")
        return global_step_training, global_step_evaluate

    global_step_training = 0
    global_step_evaluate = 0

    training_loaders = {
        3: loader_training_3,
        4: loader_training_4,
        5: loader_training_5,
    }

    for fine_class in FINE_CLASSES:
        logger.info(f"Training fine class {fine_class}")
        loader_training = training_loaders[fine_class]
        for epoch in range(NUM_EPOCHS):
            global_step_training, global_step_evaluate = do_epoch(
                epoch, 
                global_step_training, 
                global_step_evaluate,
                loader_training = loader_training,
            )


if __name__ == "__main__":
    main()