import argparse
import logging
from dataclasses import dataclass

from environment.cifar.cifar_dataset import CifarDatasetOptions
from environment.cifar.cifar_results import CifarResults

logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class CifarExperimentConfig:
    """
    The settings an experiment script takes from its command line (CifarArgs), with every default resolved, so each
    rule that turns a flag into a setting lives in one place: from_args.
    """

    CIFAR_DATA_FILE_PATH = "../cifar-100-python"
    NUM_CLASSES = 20  # coarse classes

    experiment_type:str
    batch_size:int
    learning_rate:float
    num_epochs:int  # per phase in continual
    evaluate_interval_epochs:int  # --evaluate-epochs
    max_instances:int|None
    fine_classes:list[int]
    coarse_classes:list[int]
    ltm_checkpoint:str
    seed:int|None
    run_root:str
    val_holdout:int
    split_seed:int
    # The training sets hold out the validation images and, with --max-instances and --seed, draw the subset the head
    # script draws for that seed. The validation sets are the held-out training images; None without a split.
    training_dataset_options:CifarDatasetOptions
    validation_dataset_options:CifarDatasetOptions|None

    @property
    def experiment_name(self) -> str:
        """Names the run directory: runs/cifar_100/<experiment name>/<timestamp>/."""
        max_instances_description = str(self.max_instances) if self.max_instances is not None else "500"
        return f"{self.experiment_type}_{self.fine_classes}_{max_instances_description}"

    @staticmethod
    def get_common_settings(args:argparse.Namespace, default_ltm_checkpoint:str) -> dict:
        """The settings of CifarArgs.add_common_args except --epochs, whose default differs between the scripts."""
        return dict(
            experiment_type = args.experiment_type,
            batch_size = args.batch_size,
            learning_rate = args.learning_rate,
            evaluate_interval_epochs = args.evaluate_epochs,
            max_instances = args.max_instances,
            fine_classes = args.fine_classes,
            coarse_classes = args.coarse_classes,
            ltm_checkpoint = args.ltm_checkpoint if args.ltm_checkpoint is not None else default_ltm_checkpoint,
            seed = args.seed,
            run_root = args.run_root,
            val_holdout = args.val_holdout,
            split_seed = args.split_seed,
            training_dataset_options = CifarDatasetOptions.for_training(
                args.max_instances, args.seed, args.val_holdout, args.split_seed),
            validation_dataset_options = CifarDatasetOptions.for_validation(args.val_holdout, args.split_seed),
        )


@dataclass(kw_only=True)
class StmExperimentConfig(CifarExperimentConfig):
    """Settings of cifar_main_stm_training.py. CifarAgentConfig / CifarModelConfig.from_experiment build on them."""

    LTM_CHECKPOINT = "../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth"
    STM_CHECKPOINT = "../cifar_100_pretrain/cifar_100_cc56_subclasses_12_stm_pretrain_e12.pth"
    NUM_EPOCHS = 12  # default of --epochs

    # The paper's STM, not set from the command line
    ENCODING_SIZE = 128
    BIAS_SIZE = 128
    BIAS_STAGE = 2
    CONTEXT_SIZE = 8
    DISCOUNT = 0.5
    MAX_EPISODE_STEPS = 8
    LOG_PREFIX = "cifar-100-bias"
    LOG_PERIOD = 100

    sparsity:int
    actor_training:str
    stm_checkpoint:str
    stm_checkpoint_out:str|None
    pretrain_coarse_classes:list[int]|None
    eval_bias:str
    # The episodic environment doesn't pull from the dataset until exhaustion, because all samples have to be
    # constantly fed with data which persists for multiple steps. Instead, track steps and break periodically for
    # evaluation. We still call these breaks "epochs" although they aren't true epochs in the data.
    training_steps:int  # agent steps per epoch, 4000 by default
    evaluate_steps:int  # default 1600 = 200 test images x 8 steps
    # --eval-sweep replaces evaluate_steps by one pass over every image; --eval-batch-size sets the evaluation
    # environments apart from the training batch.
    eval_sweep:bool
    eval_record_images:bool
    evaluate_batch_size:int
    ltm_obs_cache:str

    @staticmethod
    def from_args(args:argparse.Namespace) -> "StmExperimentConfig":
        if args.eval_record_images and not args.eval_sweep:
            raise ValueError("--eval-record-images needs --eval-sweep")
        return StmExperimentConfig(
            **CifarExperimentConfig.get_common_settings(args, StmExperimentConfig.LTM_CHECKPOINT),
            num_epochs = args.epochs if args.epochs is not None else StmExperimentConfig.NUM_EPOCHS,
            sparsity = args.sparsity,
            actor_training = args.actor_training,
            stm_checkpoint = args.stm_checkpoint if args.stm_checkpoint is not None else StmExperimentConfig.STM_CHECKPOINT,
            stm_checkpoint_out = args.stm_checkpoint_out,
            pretrain_coarse_classes = args.pretrain_coarse_classes,
            eval_bias = args.eval_bias,
            training_steps = args.training_steps,
            evaluate_steps = args.evaluate_steps,
            eval_sweep = args.eval_sweep,
            eval_record_images = args.eval_record_images,
            evaluate_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size,
            ltm_obs_cache = args.ltm_obs_cache,
        )


@dataclass(kw_only=True)
class LtmExperimentConfig(CifarExperimentConfig):
    """Settings of cifar_main_ltm_fine_tuning.py."""

    LTM_CHECKPOINT = "../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth"
    # Comparison steps: ( 25 epochs * batch 16 * 2000 steps ) / steps per episode 8
    # = 16x 50000 / 8
    # = 100k exposures
    # / 16 = 6250 steps
    TARGET_STEPS = 6250

    instances_per_epoch:int
    evaluate_epochs_set:set[int]|None  # --evaluate-points: log-spaced evaluation epochs within each phase
    loader_workers:int

    @staticmethod
    def from_args(args:argparse.Namespace) -> "LtmExperimentConfig":
        instances_per_epoch = args.max_instances if args.max_instances is not None else 500

        # Epochs = target_steps / instances_per_epoch
        # = 12.5 epochs @ batch size 16
        num_epochs = int(LtmExperimentConfig.TARGET_STEPS / instances_per_epoch)
        logger.info(f"Target steps: {LtmExperimentConfig.TARGET_STEPS} instances / epoch: {instances_per_epoch} so num. epochs: {num_epochs}")
        if args.epochs is not None:  # budget set by the protocol (Notes/experiments/plan.md, section 3)
            num_epochs = args.epochs
            logger.info(f"Epochs per phase set by --epochs: {num_epochs}")

        evaluate_epochs_set = None
        if args.evaluate_points is not None:
            evaluate_epochs_set = CifarResults.evaluation_epochs(num_epochs, args.evaluate_points)
            logger.info(f"Evaluating at {len(evaluate_epochs_set)} log-spaced epochs of each phase")

        return LtmExperimentConfig(
            **CifarExperimentConfig.get_common_settings(args, LtmExperimentConfig.LTM_CHECKPOINT),
            num_epochs = num_epochs,
            instances_per_epoch = instances_per_epoch,
            evaluate_epochs_set = evaluate_epochs_set,
            loader_workers = args.loader_workers,
        )
