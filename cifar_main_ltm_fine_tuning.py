import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from environment.cifar.cifar_args import CifarArgs
from environment.cifar.cifar_classifier import CifarClassifier
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_results import CifarResults
from model.resnet import ResNetConfig
from util.instrumentation import Instrumentation
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    cifar_args = CifarArgs.parse_args()

    EXPERIMENT_TYPE = cifar_args.experiment_type
    BATCH_SIZE = cifar_args.batch_size
    #SPARSITY = cifar_args.sparsity
    MAX_INSTANCES = cifar_args.max_instances
    FINE_CLASSES = cifar_args.fine_classes
    COARSE_CLASSES = cifar_args.coarse_classes
    LEARNING_RATE = cifar_args.learning_rate

    logger.info(f"Exp.:{EXPERIMENT_TYPE} Fine classes:{FINE_CLASSES} Batch size:{BATCH_SIZE} max. instances:{MAX_INSTANCES} LR: {LEARNING_RATE}")

    CIFAR_DATA_FILE_PATH = "../cifar-100-python"
    CLASSIFIER_FILE_PATH = "../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth"
    NUM_CLASSES = 20

    max_instances_description = str(MAX_INSTANCES) if MAX_INSTANCES is not None else "500"
    EXPERIMENT_NAME = f"{EXPERIMENT_TYPE}_{FINE_CLASSES}_{max_instances_description}"
    print(f"Experiment name: {EXPERIMENT_NAME}")

    run_root_path = f"cifar_100/{EXPERIMENT_NAME}"
    run_path = get_run_path(
        prefix = run_root_path, 
        path = "./runs",
    )
    create_run_path(run_path)
    data_file_path = CIFAR_DATA_FILE_PATH

    results_file = CifarResults(run_path = run_path, suffix=EXPERIMENT_TYPE)
    results_file.clear_file()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:{device}")

    MAX_STEPS_TRAINING = 0  # measure in epochs
    MAX_STEPS_EVALUATE = 0  # whole epoch

    if MAX_INSTANCES is None:
        instances_per_epoch = 500
    else:
        instances_per_epoch = MAX_INSTANCES

    # Comparison steps: ( 25 epochs * batch 16 * 2000 steps ) / steps per episode 8
    # = 16x 50000 / 8 
    # = 100k exposures
    # / 16 = 6250 steps
    target_steps = 6250

    # Epochs = target_steps / instances_per_epoch 
    # = 12.5 epochs @ batch size 16
    NUM_EPOCHS = int(target_steps / instances_per_epoch)
    logger.info(f"Target steps: {target_steps} instances / epoch: {instances_per_epoch} so num. epochs: {NUM_EPOCHS}")

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

    dataset_training_3 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_3,
        max_instances = MAX_INSTANCES,
        as_tensor=True,
    )
    dataset_training_4 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_4,
        max_instances = MAX_INSTANCES,
        as_tensor=True,
    )
    dataset_training_5 = Cifar100Dataset(
        file_path=data_file_path, 
        label_type=Cifar100Dataset.LABEL_TYPE_COARSE,
        training=True,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine_5,
        max_instances = MAX_INSTANCES,
        as_tensor=True,
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
    loader_training_3 = DataLoader(
        dataset_training_3,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    loader_training_4 = DataLoader(
        dataset_training_4,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )
    loader_training_5 = DataLoader(
        dataset_training_5,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=pin_memory,
    )

    loader_evaluate_12 = DataLoader(
        dataset_evaluate_12,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    loader_evaluate_3 = DataLoader(
        dataset_evaluate_3,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    loader_evaluate_4 = DataLoader(
        dataset_evaluate_4,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )
    loader_evaluate_5 = DataLoader(
        dataset_evaluate_5,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    config = ResNetConfig(
        num_classes=NUM_CLASSES,
    )
    model = CifarClassifier(config, bias_stage=-1).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"# parameters:{num_parameters}")

    logger.info(f"Loading conv. model from file: {CLASSIFIER_FILE_PATH}")
    state_dict = torch.load(CLASSIFIER_FILE_PATH, weights_only=True)
    model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            model.train()
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
        results_file.append_line(
            coarse_classes = COARSE_CLASSES,
            fine_classes = FINE_CLASSES,
            mode = Instrumentation.MODE_TRAINING,
            epoch = epoch,
            accuracy = metrics_training.mean_accuracy,            
        )

        evaluate_names = ["12","3","4","5"]
        evaluate_loaders = [loader_evaluate_12, loader_evaluate_3, loader_evaluate_4, loader_evaluate_5]
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