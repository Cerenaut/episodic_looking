
import logging
import math
import os
from dataclasses import dataclass

import gymnasium as gym
import torch

from environment.cifar.cifar_agent import CifarAgent, CifarAgentConfig
from environment.cifar.cifar_args import CifarArgs
from environment.cifar.cifar_dataset import Cifar100Dataset, CifarDatasetOptions
from environment.cifar.cifar_env import CifarEnv, CifarEnvConfig, EvaluationSweep
from environment.cifar.cifar_experiment import StmExperimentConfig
from environment.cifar.cifar_model import CifarModelConfig
from environment.cifar.cifar_results import CifarResults
from util.device import get_device, seed_all
from util.instrumentation import Instrumentation
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def do_steps_in_mode(agent:CifarAgent, num_steps:int, mode:str):
    CifarEnv.set_mode_for_envs(
        envs = agent.get_envs(mode),
        mode = mode,
    )
    agent.set_mode(mode)  # also calls reset on agent and envs
    agent.reset_cumulative_accuracy()
    agent.do_steps(num_steps=num_steps)
    accuracy = agent.get_cumulative_accuracy()
    logger.info(f"Mode:{mode} Accuracy:{accuracy}")
    return accuracy

def do_training(agent, num_steps:int) -> float:
    return do_steps_in_mode(
        agent = agent,
        num_steps = num_steps, 
        mode = Instrumentation.MODE_TRAINING, 
    )

def do_evaluate(agent, num_steps:int) -> float:
    # Adjust eval log period so we can see accuracy across episode
    #temp = agent.instrumentation.config.log_period
    #agent.instrumentation.config.log_period = 1  # write eval results every step
    accuracy = do_steps_in_mode(
        agent = agent,
        num_steps = num_steps, 
        mode = Instrumentation.MODE_EVALUATE, 
    )
    #agent.instrumentation.config.log_period = temp
    return accuracy    

def do_evaluate_sweep(agent:CifarAgent, episode_steps:int) -> tuple[float, dict[int, int]]:
    """
    One deterministic pass over every image of the current evaluation dataset, one episode per image, at the
    evaluation batch size. Returns the accuracy and each image's result (dataset index -> 1/0).
    """
    mode = Instrumentation.MODE_EVALUATE
    envs = agent.get_envs(mode)
    CifarEnv.set_mode_for_envs(envs = envs, mode = mode)
    num_images = envs.call("get_num_images")[0]
    CifarEnv.set_image_sweep_for_envs(envs, EvaluationSweep(num_images))
    agent.set_mode(mode)  # resets the environments, which take the sweep's first images
    agent.reset_cumulative_accuracy()
    agent.start_sweep()
    num_rounds = math.ceil(num_images / agent.get_batch_size(mode))
    agent.do_steps(num_steps = num_rounds * episode_steps)
    image_correct = agent.stop_sweep()
    CifarEnv.set_image_sweep_for_envs(envs, None)
    if sorted(image_correct) != list(range(num_images)):
        raise RuntimeError(f"Evaluation sweep scored {len(image_correct)} of {num_images} images")
    accuracy = agent.get_cumulative_accuracy()
    logger.info(f"Mode:{mode} sweep of {num_images} images Accuracy:{accuracy}")
    return accuracy, image_correct

@dataclass
class StmRun:
    """
    What every experiment type shares: the settings, the agent, the class filters of the environments' first datasets
    and the results files.
    """
    experiment:StmExperimentConfig
    agent:CifarAgent
    exclude_classes_coarse:set[int]
    exclude_classes_fine:set[int]|None  # all fine classes but --fine-classes
    dataset_evaluate:Cifar100Dataset  # the environments' first evaluation dataset: the test images of --fine-classes
    results_file:CifarResults
    results_file_val:CifarResults|None  # same format and epochs as results_file, on the validation sets
    results_file_images:CifarResults|None  # per-image results of every evaluation sweep

    def create_dataset(
        self,
        mode:str,
        exclude_classes_fine:set[int]|None,
        max_instances:int|None,
        dataset_options:CifarDatasetOptions|None = None,
    ) -> Cifar100Dataset:
        """A dataset of the --coarse-classes, in new shared memory that the environments can attach to."""
        return CifarEnv.create_dataset(
            data_file_path = self.experiment.CIFAR_DATA_FILE_PATH,
            mode = mode,
            shared_memory_names = None,
            exclude_classes_coarse = self.exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine,
            max_instances = max_instances,
            dataset_options = dataset_options,
        )

    def train_on(self, exclude_classes_fine:set[int]|None, description:str):
        """Switch the training environments to the training images of the fine classes not excluded."""
        experiment = self.experiment
        dataset_training = self.create_dataset(
            mode = Instrumentation.MODE_TRAINING,
            exclude_classes_fine = exclude_classes_fine,
            max_instances = experiment.max_instances,
            dataset_options = experiment.training_dataset_options,
        )
        logger.info(f"{description}: {len(dataset_training)} images")

        CifarEnv.set_dataset_config_for_envs(
            envs = self.agent.get_envs(Instrumentation.MODE_TRAINING),
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = dataset_training.get_shared_memory_names(),
            exclude_classes_fine = exclude_classes_fine,
            max_instances = experiment.max_instances,  # as dataset_training (None: all); continual was given None, which failed
            dataset_options = experiment.training_dataset_options,
        )

    def train_epoch(self) -> float:
        return do_training(agent = self.agent, num_steps = self.experiment.training_steps)

    def append_training_line(self, fine_classes:list[int], epoch:int, accuracy:float, coarse_classes:list[int]|None = None):
        # The training line goes into the validation file too, so both files have the head script's layout.
        if coarse_classes is None:
            coarse_classes = self.experiment.coarse_classes
        for results in (self.results_file, self.results_file_val):
            if results is None:
                continue
            results.append_line(
                coarse_classes = str(coarse_classes),
                fine_classes = str(fine_classes),
                mode = Instrumentation.MODE_TRAINING,
                epoch = epoch,
                accuracy = accuracy,
            )

    def evaluate_current(self, record_label:str, epoch:int) -> float:
        """Evaluate the environments' current evaluation dataset, by sweep or by --evaluate-steps steps."""
        if not self.experiment.eval_sweep:
            return do_evaluate(agent = self.agent, num_steps = self.experiment.evaluate_steps)
        accuracy, image_correct = do_evaluate_sweep(self.agent, self.experiment.MAX_EPISODE_STEPS)
        if self.results_file_images is not None:
            bits = "".join(str(image_correct[i]) for i in range(len(image_correct)))
            self.results_file_images.append_file(f"{record_label}, {epoch}, {bits}\n")
        return accuracy

    def save_stm(self, file_path:str, description:str):
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(self.agent.model.state_dict(), file_path)
        logger.info(f"Wrote STM {description} to: {file_path}")


@dataclass
class EvaluationGroup:
    """Fine classes evaluated together: their test set and, with the validation split, their validation set."""
    name:str  # in the results files, e.g. "['12']"
    exclude_classes_fine:set[int]|None
    dataset_test:Cifar100Dataset
    dataset_validation:Cifar100Dataset|None = None


def evaluate_groups(run:StmRun, groups:list[EvaluationGroup], epoch:int):
    """Evaluate every group's test set, into the results file, then every group's validation set, into its own."""
    for part, results in (("test", run.results_file), ("validation", run.results_file_val)):
        dataset_options = None if part == "test" else run.experiment.validation_dataset_options
        for group in groups:
            dataset = group.dataset_test if part == "test" else group.dataset_validation
            if dataset is None:
                continue
            logger.info(f"Evaluating ({part}) @ epoch {epoch} fine classes {group.name}...")
            CifarEnv.set_dataset_config_for_envs(
                envs = run.agent.get_envs(Instrumentation.MODE_EVALUATE),
                mode = Instrumentation.MODE_EVALUATE,
                shared_memory_names = dataset.get_shared_memory_names(),
                exclude_classes_fine = group.exclude_classes_fine,
                max_instances = None,  # All instances
                dataset_options = dataset_options,
            )
            accuracy_evaluate = run.evaluate_current(f"{part}, {group.name}", epoch)
            results.append_line(
                coarse_classes = str(run.experiment.coarse_classes),
                fine_classes = group.name,
                mode = Instrumentation.MODE_EVALUATE,
                epoch = epoch,
                accuracy = accuracy_evaluate,
            )


def create_run(experiment:StmExperimentConfig) -> StmRun:
    """Construct all the objects needed for the experiment, including model and envs, and the results files."""
    EXPERIMENT_TYPE = experiment.experiment_type
    FINE_CLASSES = experiment.fine_classes
    COARSE_CLASSES = experiment.coarse_classes
    CIFAR_DATA_FILE_PATH = experiment.CIFAR_DATA_FILE_PATH

    run_root_path = f"cifar_100/{experiment.experiment_name}"
    run_path = get_run_path(
        prefix=run_root_path,
        path=experiment.run_root,
    )
    create_run_path(run_path)
    logger.info(f"Run path: {run_path}")

    results_file = CifarResults(run_path = run_path, suffix=EXPERIMENT_TYPE)
    results_file.clear_file()
    results_file_val = None
    if experiment.val_holdout > 0:
        results_file_val = CifarResults(run_path = run_path, suffix=f"{EXPERIMENT_TYPE}_val")
        results_file_val.clear_file()
    results_file_images = None
    if experiment.eval_record_images:
        results_file_images = CifarResults(run_path = run_path, suffix=f"{EXPERIMENT_TYPE}_images")
        results_file_images.clear_file()

    device = get_device()  # cuda > mps > cpu; override with EPISODIC_DEVICE env var. Same choice CifarAgent makes internally.
    logger.info(f"Device: {device}")

    # We need a base dataset to initialize the model and environments.
    # Pre-create shared memory for all the dataset subsets used (training, evaluate, various sub-class filters)
    image_shape = Cifar100Dataset.get_image_shape()

    exclude_classes_fine = []
    all_fine_classes = [1,2,3,4,5]
    for fine_class in all_fine_classes:
        if fine_class not in FINE_CLASSES:
            exclude_classes_fine.append(fine_class)
    exclude_classes_fine = Cifar100Dataset.get_fine_classes(exclude_classes_fine)
    logger.info(
        f"Using fine classes: {FINE_CLASSES} excluding: {exclude_classes_fine}"
    )

    exclude_classes_coarse = Cifar100Dataset.get_coarse_classes_excluded(COARSE_CLASSES)
    logger.info(
        f"Using coarse classes: {COARSE_CLASSES} excluding: {exclude_classes_coarse}"
    )
    # D.15: pre-train on the fine classes of a broader set of coarse classes; evaluation stays on COARSE_CLASSES.
    exclude_classes_coarse_training = None
    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_PRETRAIN and experiment.pretrain_coarse_classes is not None:
        exclude_classes_coarse_training = Cifar100Dataset.get_coarse_classes_excluded(experiment.pretrain_coarse_classes)
        logger.info(f"Pre-training on coarse classes: {experiment.pretrain_coarse_classes} (evaluation on {COARSE_CLASSES})")

    dataset_training = CifarEnv.create_dataset(
        data_file_path=CIFAR_DATA_FILE_PATH,
        mode=Instrumentation.MODE_TRAINING,
        shared_memory_names=None,
        exclude_classes_coarse=exclude_classes_coarse if exclude_classes_coarse_training is None else exclude_classes_coarse_training,
        exclude_classes_fine=exclude_classes_fine,
        max_instances=experiment.max_instances,
        dataset_options=experiment.training_dataset_options,
    )

    dataset_evaluate = CifarEnv.create_dataset(
        data_file_path=CIFAR_DATA_FILE_PATH,
        mode=Instrumentation.MODE_EVALUATE,
        shared_memory_names=None,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine,
        max_instances=None,
    )

    env_config = CifarEnvConfig(
        data_file_path=CIFAR_DATA_FILE_PATH,
        image_shape=image_shape,
        num_classes=experiment.NUM_CLASSES,
        max_steps=experiment.MAX_EPISODE_STEPS,
        shared_memory_names_training=dataset_training.get_shared_memory_names(),
        shared_memory_names_evaluate=dataset_evaluate.get_shared_memory_names(),
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_coarse_training=exclude_classes_coarse_training,
        exclude_classes_fine_training=exclude_classes_fine,
        exclude_classes_fine_evaluate=exclude_classes_fine,
        dataset_options_training=experiment.training_dataset_options,
        max_instances_training=experiment.max_instances,  # as dataset_training, which the environments attach to
    )

    class NoArgCifarEnv(CifarEnv):
        ENV_NAME = "CifarEnv-v0"

        def __init__(
            self,
        ):
            super().__init__(env_config)

    gym.register(
        id=NoArgCifarEnv.ENV_NAME,
        entry_point=NoArgCifarEnv,
    )

    agent = CifarAgent(
        config=CifarAgentConfig.from_experiment(
            experiment = experiment,
            environment_id = NoArgCifarEnv.ENV_NAME,
            image_shape = image_shape,
        ),
        model_config=CifarModelConfig.from_experiment(experiment),
    )
    if experiment.seed is not None:
        for envs in {id(e): e for e in (agent.get_envs(Instrumentation.MODE_TRAINING), agent.get_envs(Instrumentation.MODE_EVALUATE))}.values():
            envs.action_space.seed(experiment.seed)  # only used by the random-action path of the agent
            envs.single_action_space.seed(experiment.seed)

    return StmRun(
        experiment = experiment,
        agent = agent,
        exclude_classes_coarse = exclude_classes_coarse,
        exclude_classes_fine = exclude_classes_fine,
        dataset_evaluate = dataset_evaluate,
        results_file = results_file,
        results_file_val = results_file_val,
        results_file_images = results_file_images,
    )


def run_pretrain(run:StmRun):
    """Train the STM on --fine-classes (1 2), evaluate on their test (and validation) images, save the STM."""
    experiment = run.experiment
    # Training rows record the coarse classes the training accuracy was measured over (D.15: the broader set).
    coarse_classes_training = experiment.coarse_classes if experiment.pretrain_coarse_classes is None else experiment.pretrain_coarse_classes

    group = EvaluationGroup(
        name = str(experiment.fine_classes),
        exclude_classes_fine = run.exclude_classes_fine,
        dataset_test = run.dataset_evaluate,
    )
    if experiment.validation_dataset_options is not None:
        group.dataset_validation = run.create_dataset(
            mode = Instrumentation.MODE_EVALUATE,
            exclude_classes_fine = run.exclude_classes_fine,
            max_instances = None,
            dataset_options = experiment.validation_dataset_options,
        )

    evaluate_interval_epochs = experiment.evaluate_interval_epochs
    for epoch in range(experiment.num_epochs):
        accuracy_training = run.train_epoch()
        run.append_training_line(experiment.fine_classes, epoch, accuracy_training, coarse_classes = coarse_classes_training)

        if evaluate_interval_epochs > 1 and (epoch % evaluate_interval_epochs != 0):
            continue  # Option to skip evals during pretraining, useful with small batch size

        evaluate_groups(run, [group], epoch)

    # Test save / load STM
    run.save_stm(experiment.stm_checkpoint, "after pre-training")


def load_stm(run:StmRun):
    file_path = run.experiment.stm_checkpoint
    logger.info(f"Reading STM from: {file_path}")
    state_dict = torch.load(file_path, weights_only=True, map_location="cpu")  # checkpoints move between CUDA/MPS machines
    run.agent.model.load_state_dict(state_dict)


# Test (and validation) sets per group of fine classes: 1,2 together, as they were pre-trained, then 3, 4, 5.
EXCLUDE_CLASSES_FINE_EVALUATE_BY_KEY = {
    "12": Cifar100Dataset.get_fine_classes([    3,4,5,]),
    "3":  Cifar100Dataset.get_fine_classes([1,2,  4,5,]),
    "4":  Cifar100Dataset.get_fine_classes([1,2,3,  5,]),
    "5":  Cifar100Dataset.get_fine_classes([1,2,3,4,  ]),
}


def create_evaluation_groups(run:StmRun) -> list[EvaluationGroup]:
    """
    Create shared copies of the evaluate dataset for fine classes 1-5, 1 and 2 grouped together as they were
    pre-trained, and with the validation split, of the validation sets (held-out training images).
    """
    groups = []
    for key, exclude_classes_fine_evaluate in EXCLUDE_CLASSES_FINE_EVALUATE_BY_KEY.items():
        groups.append(EvaluationGroup(
            name = str([key]),
            exclude_classes_fine = exclude_classes_fine_evaluate,
            dataset_test = run.create_dataset(
                mode = Instrumentation.MODE_EVALUATE,
                exclude_classes_fine = exclude_classes_fine_evaluate,
                max_instances = None,  # For evaluate, always all
            ),
        ))
    logger.info("Eval. individual fine classes using datasets:")
    for group in groups:
        logger.info(f"Fine classes: {group.name} --> shared memory: {group.dataset_test.shared_memory_names} size: {len(group.dataset_test)}")

    if run.experiment.validation_dataset_options is not None:
        for group in groups:
            group.dataset_validation = run.create_dataset(
                mode = Instrumentation.MODE_EVALUATE,
                exclude_classes_fine = group.exclude_classes_fine,
                max_instances = None,
                dataset_options = run.experiment.validation_dataset_options,
            )
            logger.info(f"Validation, fine classes: {group.name} --> size: {len(group.dataset_validation)}")
    return groups


def run_evaluate(run:StmRun, groups:list[EvaluationGroup]):
    """Evaluate the loaded STM checkpoint on the four test (and validation) sets once, no training."""
    evaluate_groups(run, groups, epoch=0)


def run_few_shot(run:StmRun, groups:list[EvaluationGroup]):
    """
    Train on --max-instances images per coarse class of one fine class, 3, 4 or 5 (single-stream: all of them at
    batch 1), and evaluate all five, reported separately.
    """
    experiment = run.experiment
    FINE_CLASSES = experiment.fine_classes
    exclude_classes_fine_few_shot_training = None
    if FINE_CLASSES == [3]:
        exclude_classes_fine_few_shot_training = Cifar100Dataset.get_fine_classes([1,2, 4,5,])
    elif FINE_CLASSES == [4]:
        exclude_classes_fine_few_shot_training = Cifar100Dataset.get_fine_classes([1,2,3, 5,])
    elif FINE_CLASSES == [5]:
        exclude_classes_fine_few_shot_training = Cifar100Dataset.get_fine_classes([1,2,3,4, ])
    run.train_on(exclude_classes_fine_few_shot_training, "Few-shot training set")

    num_epochs = experiment.num_epochs
    evaluate_interval_epochs = experiment.evaluate_interval_epochs
    for epoch in range(num_epochs):
        accuracy_training = run.train_epoch()
        run.append_training_line(FINE_CLASSES, epoch, accuracy_training)

        is_last_epoch = (epoch == num_epochs - 1)
        if evaluate_interval_epochs > 1 and (epoch % evaluate_interval_epochs != 0) and not is_last_epoch:
            continue  # Option to skip evals during pretraining, useful with small batch size

        # Always evaluate on the final epoch, so the last measured point sits at the full
        # training budget rather than at the last multiple of evaluate_interval_epochs.
        evaluate_groups(run, groups, epoch)

    if experiment.stm_checkpoint_out is not None:
        run.save_stm(experiment.stm_checkpoint_out, "after few-shot training")


def run_continual(run:StmRun, groups:list[EvaluationGroup]):
    """Train on fine classes 3, 4, 5 in the order of --fine-classes, num_epochs each, evaluating every epoch."""
    experiment = run.experiment
    continual_exclusions = {
        3: Cifar100Dataset.get_fine_classes([1,2, 4,5,]),
        4: Cifar100Dataset.get_fine_classes([1,2,3, 5,]),
        5: Cifar100Dataset.get_fine_classes([1,2,3,4, ]),
    }

    start_epoch = 0
    for fine_class in experiment.fine_classes:
        logger.info(f"Training fine class {fine_class}")
        run.train_on(continual_exclusions[fine_class], f"Continual training set, fine class {[fine_class]}")

        for epoch in range(experiment.num_epochs):
            accuracy_training = run.train_epoch()
            run.append_training_line([fine_class], epoch, accuracy_training)
            evaluate_groups(run, groups, start_epoch + epoch)
        start_epoch += experiment.num_epochs

        if experiment.stm_checkpoint_out is not None:
            run.save_stm(f"{experiment.stm_checkpoint_out}_phase{fine_class}.pth", f"after fine class {fine_class}")

        steps_complete_training = run.agent.instrumentation.get_steps(mode = Instrumentation.MODE_TRAINING)
        steps_complete_evaluate = run.agent.instrumentation.get_steps(mode = Instrumentation.MODE_EVALUATE)    
        logger.info(f"Fine class {fine_class} complete @ {steps_complete_training} (training) / {steps_complete_evaluate} (evaluate)")


def main():
    args = CifarArgs.parse_stm_args()
    experiment = StmExperimentConfig.from_args(args)

    if experiment.seed is not None:
        # Model init and policy sampling (torch), env image sampling and dataset subsets (numpy global RNG,
        # envs are synchronous so they share it), python's random for completeness.
        seed_all(experiment.seed)

    logger.info(f"Exp.:{experiment.experiment_type} Fine classes:{experiment.fine_classes} Batch size:{experiment.batch_size} max. instances:{experiment.max_instances} LR: {experiment.learning_rate} Seed: {experiment.seed}")
    # Validation split (--val-holdout): every training set holds out the same images per fine class as the head
    # script, and the held-out images are evaluated alongside the test sets.
    # With --max-instances and --seed, the subset is the one the head and LTM-only scripts draw for that seed
    # (Cifar100Dataset.get_subset_mask), and environments re-derive it identically; without a seed, the old draw.
    if experiment.training_dataset_options.subset_seed is not None:
        logger.info(f"Subset of {experiment.max_instances} images per coarse class drawn with seed {experiment.seed}")
    if experiment.validation_dataset_options is not None:
        logger.info(f"Validation split: {experiment.val_holdout} images per fine class held out, split seed {experiment.split_seed}")
    logger.info(f"LTM checkpoint: {experiment.ltm_checkpoint} STM checkpoint: {experiment.stm_checkpoint} actor training: {experiment.actor_training}")
    logger.info(f"Evaluation: {'one pass over every image' if experiment.eval_sweep else f'{experiment.evaluate_steps} steps'}, "
                f"{experiment.evaluate_batch_size} environments")

    run = create_run(experiment)

    EXPERIMENT_TYPE = experiment.experiment_type
    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_PRETRAIN:
        run_pretrain(run)
        return

    load_stm(run)
    groups = create_evaluation_groups(run)
    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_EVALUATE:
        run_evaluate(run, groups)
    elif EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_FEW_SHOT:
        run_few_shot(run, groups)
    elif EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_CONTINUAL:
        run_continual(run, groups)
    # streaming: no code path of its own (single-stream runs use few-shot at batch 1)

if __name__ == "__main__":
    main()
