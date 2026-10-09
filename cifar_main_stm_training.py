
import logging
import math
import os

import gymnasium as gym
import torch

from environment.cifar.cifar_agent import CifarAgent, CifarAgentConfig
from environment.cifar.cifar_args import CifarArgs
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_env import CifarEnv, CifarEnvConfig, EvaluationSweep
from environment.cifar.cifar_model import CifarModel, CifarModelConfig
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

def main():
    args = CifarArgs.parse_args()

    SEED = args.seed
    if SEED is not None:
        # Model init and policy sampling (torch), env image sampling and dataset subsets (numpy global RNG,
        # envs are synchronous so they share it), python's random for completeness.
        seed_all(SEED)

    EXPERIMENT_TYPE = args.experiment_type
    BATCH_SIZE = args.batch_size
    SPARSITY = args.sparsity
    MAX_INSTANCES = args.max_instances
    FINE_CLASSES = args.fine_classes
    COARSE_CLASSES = args.coarse_classes
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.epochs if args.epochs is not None else 12  # --epochs defaults to None, for LTM-only's rule
    EVALUATE_INTERVAL_EPOCHS = args.evaluate_epochs
    VAL_HOLDOUT = args.val_holdout
    logger.info(f"Exp.:{EXPERIMENT_TYPE} Fine classes:{FINE_CLASSES} Batch size:{BATCH_SIZE} max. instances:{MAX_INSTANCES} LR: {LEARNING_RATE} Seed: {SEED}")

    # Validation split (--val-holdout): every training set holds out the same images per fine class as the head
    # script, and the held-out images are evaluated alongside the test sets.
    # With --max-instances and --seed, the subset is the one the head and LTM-only scripts draw for that seed
    # (Cifar100Dataset.get_subset_mask), and environments re-derive it identically; without a seed, the old draw.
    subset_options = Cifar100Dataset.get_subset_options(MAX_INSTANCES, SEED)
    split_options = Cifar100Dataset.get_split_options(VAL_HOLDOUT, args.split_seed)
    TRAINING_DATASET_OPTIONS = {**subset_options, **split_options} or None
    VALIDATION_DATASET_OPTIONS = None  # "training": the validation set is read from the training file
    if subset_options:
        logger.info(f"Subset of {MAX_INSTANCES} images per coarse class drawn with seed {SEED}")
    if split_options:
        VALIDATION_DATASET_OPTIONS = {"training": True, **split_options, "split_part": "validation"}
        logger.info(f"Validation split: {VAL_HOLDOUT} images per fine class held out, split seed {args.split_seed}")

    LOG_PREFIX = "cifar-100-bias"
    LOG_PERIOD = 100

    NUM_CLASSES = 20  # Coarse classes
    ENCODING_SIZE = 128
    BIAS_SIZE = 128
    BIAS_STAGE = 2

    CONTEXT_SIZE = 8
    DISCOUNT = 0.5
    MAX_EPISODE_STEPS = 8

    CIFAR_DATA_FILE_PATH = "../cifar-100-python"
    CLASSIFIER_FILE_PATH = "../cifar_100_pretrain/cifar_100_subclasses_12_e11_31.1.pth"
    STM_PRETRAIN_FILE_PATH = "../cifar_100_pretrain/cifar_100_cc56_subclasses_12_stm_pretrain_e12.pth"
    if args.ltm_checkpoint is not None:
        CLASSIFIER_FILE_PATH = args.ltm_checkpoint
    if args.stm_checkpoint is not None:
        STM_PRETRAIN_FILE_PATH = args.stm_checkpoint
    ACTOR_TRAINING = args.actor_training
    logger.info(f"LTM checkpoint: {CLASSIFIER_FILE_PATH} STM checkpoint: {STM_PRETRAIN_FILE_PATH} actor training: {ACTOR_TRAINING}")

    # The episodic environment doesn't pull from the dataset until exhaustion, because all samples
    # have to be constantly fed with data which persists for multiple steps.
    # Instead, track steps and break periodically for evaluation. We still call these breaks "epochs"
    # although they aren't true epochs in the data.
    TRAINING_STEPS = args.training_steps  # 4000 by default
    EVALUATE_STEPS = args.evaluate_steps  # default 1600 = 200 test images x 8 steps
    # --eval-sweep replaces EVALUATE_STEPS by one pass over every image; --eval-batch-size sets the evaluation
    # environments apart from the training batch.
    EVAL_SWEEP = args.eval_sweep
    EVALUATE_BATCH_SIZE = args.eval_batch_size if args.eval_batch_size is not None else BATCH_SIZE
    logger.info(f"Evaluation: {'one pass over every image' if EVAL_SWEEP else f'{EVALUATE_STEPS} steps'}, "
                f"{EVALUATE_BATCH_SIZE} environments")
    if args.eval_record_images and not EVAL_SWEEP:
        raise ValueError("--eval-record-images needs --eval-sweep")

    # Construct all the objects needed for the experiment, including model and envs
    max_instances_description = (
        str(MAX_INSTANCES) if MAX_INSTANCES is not None else "500"
    )
    EXPERIMENT_NAME = f"{EXPERIMENT_TYPE}_{FINE_CLASSES}_{max_instances_description}"
    run_root_path = f"cifar_100/{EXPERIMENT_NAME}"
    run_path = get_run_path(
        prefix=run_root_path,
        path=args.run_root,
    )
    create_run_path(run_path)
    logger.info(f"Run path: {run_path}")

    results_file = CifarResults(run_path = run_path, suffix=EXPERIMENT_TYPE)
    results_file.clear_file()
    results_file_val = None  # same format and epochs as results_file, on the validation sets
    if VAL_HOLDOUT > 0:
        results_file_val = CifarResults(run_path = run_path, suffix=f"{EXPERIMENT_TYPE}_val")
        results_file_val.clear_file()
    results_file_images = None  # per-image results of every evaluation sweep
    if args.eval_record_images:
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
    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_PRETRAIN and args.pretrain_coarse_classes is not None:
        exclude_classes_coarse_training = Cifar100Dataset.get_coarse_classes_excluded(args.pretrain_coarse_classes)
        logger.info(f"Pre-training on coarse classes: {args.pretrain_coarse_classes} (evaluation on {COARSE_CLASSES})")

    dataset_training = CifarEnv.create_dataset(
        data_file_path=CIFAR_DATA_FILE_PATH,
        mode=Instrumentation.MODE_TRAINING,
        shared_memory_names=None,
        exclude_classes_coarse=exclude_classes_coarse if exclude_classes_coarse_training is None else exclude_classes_coarse_training,
        exclude_classes_fine=exclude_classes_fine,
        max_instances=MAX_INSTANCES,
        dataset_options=TRAINING_DATASET_OPTIONS,
    )

    dataset_evaluate = CifarEnv.create_dataset(
        data_file_path=CIFAR_DATA_FILE_PATH,
        mode=Instrumentation.MODE_EVALUATE,
        shared_memory_names=None,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine,
        max_instances=None,
    )

    shared_memory_names_training = dataset_training.get_shared_memory_names()
    shared_memory_names_evaluate = dataset_evaluate.get_shared_memory_names()

    env_config = CifarEnvConfig(
        data_file_path=CIFAR_DATA_FILE_PATH,
        image_shape=image_shape,
        num_classes=NUM_CLASSES,
        max_steps=MAX_EPISODE_STEPS,
        shared_memory_names_training=shared_memory_names_training,
        shared_memory_names_evaluate=shared_memory_names_evaluate,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_coarse_training=exclude_classes_coarse_training,
        exclude_classes_fine_training=exclude_classes_fine,
        exclude_classes_fine_evaluate=exclude_classes_fine,
        dataset_options_training=TRAINING_DATASET_OPTIONS,
        max_instances_training=MAX_INSTANCES,  # as dataset_training, which the environments attach to
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

    config = CifarAgentConfig(
        # Logging
        log_type="tensorboard",
        log_path=args.run_root,
        log_prefix=LOG_PREFIX,
        log_period=LOG_PERIOD,
        # EpisodicAgent
        batch_size=BATCH_SIZE,
        evaluate_batch_size=EVALUATE_BATCH_SIZE if EVALUATE_BATCH_SIZE != BATCH_SIZE else 0,
        history_size=CONTEXT_SIZE,
        action_size=NUM_CLASSES,
        observation_size=ENCODING_SIZE + BIAS_SIZE + NUM_CLASSES,
        random_policy=False,
        environment_id=NoArgCifarEnv.ENV_NAME,
        async_env=False,
        max_episode_steps=MAX_EPISODE_STEPS,
        # CifarAgent
        image_shape=image_shape,
        learning_rate=LEARNING_RATE,
        momentum=0.5,
        ltm_obs_cache=args.ltm_obs_cache,
    )

    model_config = CifarModelConfig(
        encoder_ensemble_size=1,
        encoder_sparsity=SPARSITY,
        history_size=CONTEXT_SIZE,
        model_hidden_size=1000,
        model_hidden_size_factor=1,
        model_nonlinearity="leaky-relu",
        model_layers=3,
        discount_factor=DISCOUNT,
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
        classifier_model_file=CLASSIFIER_FILE_PATH,
        classifier_bias_stage=BIAS_STAGE,
        policy_std=0.5,
        actor_training=ACTOR_TRAINING,
        eval_bias=args.eval_bias,
    )

    agent = CifarAgent(
        config=config,
        model_config=model_config,
    )
    if SEED is not None:
        for envs in {id(e): e for e in (agent.get_envs(Instrumentation.MODE_TRAINING), agent.get_envs(Instrumentation.MODE_EVALUATE))}.values():
            envs.action_space.seed(SEED)  # only used by the random-action path of the agent
            envs.single_action_space.seed(SEED)

    def evaluate_current(record_label:str, epoch:int) -> float:
        """Evaluate the environments' current evaluation dataset, by sweep or by EVALUATE_STEPS steps."""
        if not EVAL_SWEEP:
            return do_evaluate(agent = agent, num_steps = EVALUATE_STEPS)
        accuracy, image_correct = do_evaluate_sweep(agent, MAX_EPISODE_STEPS)
        if results_file_images is not None:
            bits = "".join(str(image_correct[i]) for i in range(len(image_correct)))
            results_file_images.append_file(f"{record_label}, {epoch}, {bits}\n")
        return accuracy

    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_PRETRAIN:
        # Training rows record the coarse classes the training accuracy was measured over (D.15: the broader set).
        coarse_classes_training = COARSE_CLASSES if args.pretrain_coarse_classes is None else args.pretrain_coarse_classes

        dataset_validation_pretrain = None
        if VAL_HOLDOUT > 0:
            dataset_validation_pretrain = CifarEnv.create_dataset(
                data_file_path=CIFAR_DATA_FILE_PATH,
                mode=Instrumentation.MODE_EVALUATE,
                shared_memory_names=None,
                exclude_classes_coarse=exclude_classes_coarse,
                exclude_classes_fine=exclude_classes_fine,
                max_instances=None,
                dataset_options=VALIDATION_DATASET_OPTIONS,
            )

        for epoch in range(NUM_EPOCHS):
            accuracy_training = do_training(agent = agent, num_steps = TRAINING_STEPS)
            for results in (results_file, results_file_val):
                if results is None:
                    continue
                results.append_line(
                    coarse_classes = str(coarse_classes_training),
                    fine_classes = str(FINE_CLASSES),
                    mode = Instrumentation.MODE_TRAINING,
                    epoch = epoch,
                    accuracy = accuracy_training,            
                )

            if EVALUATE_INTERVAL_EPOCHS > 1 and (epoch % EVALUATE_INTERVAL_EPOCHS != 0):
                continue  # Option to skip evals during pretraining, useful with small batch size

            if dataset_validation_pretrain is not None:
                # The environments' evaluation dataset is switched between the test and validation sets.
                CifarEnv.set_dataset_config_for_envs(
                    envs = agent.get_envs(Instrumentation.MODE_EVALUATE),
                    mode = Instrumentation.MODE_EVALUATE,
                    shared_memory_names = shared_memory_names_evaluate,
                    exclude_classes_fine = exclude_classes_fine,
                    max_instances = None,
                )
            accuracy_evaluate = evaluate_current(f"test, {FINE_CLASSES}", epoch)
            results_file.append_line(
                coarse_classes = str(COARSE_CLASSES),
                fine_classes = str(FINE_CLASSES),
                mode = Instrumentation.MODE_EVALUATE,
                epoch = epoch,
                accuracy = accuracy_evaluate,            
            )

            if dataset_validation_pretrain is not None:
                CifarEnv.set_dataset_config_for_envs(
                    envs = agent.get_envs(Instrumentation.MODE_EVALUATE),
                    mode = Instrumentation.MODE_EVALUATE,
                    shared_memory_names = dataset_validation_pretrain.get_shared_memory_names(),
                    exclude_classes_fine = exclude_classes_fine,
                    max_instances = None,
                    dataset_options = VALIDATION_DATASET_OPTIONS,
                )
                accuracy_validation = evaluate_current(f"validation, {FINE_CLASSES}", epoch)
                results_file_val.append_line(
                    coarse_classes = str(COARSE_CLASSES),
                    fine_classes = str(FINE_CLASSES),
                    mode = Instrumentation.MODE_EVALUATE,
                    epoch = epoch,
                    accuracy = accuracy_validation,
                )

        # Test save / load STM
        logger.info(f"Writing STM to: {STM_PRETRAIN_FILE_PATH}")
        os.makedirs(os.path.dirname(os.path.abspath(STM_PRETRAIN_FILE_PATH)), exist_ok=True)
        torch.save(agent.model.state_dict(), STM_PRETRAIN_FILE_PATH)
        return  # End of this experiment type
    else:
        logger.info(f"Reading STM from: {STM_PRETRAIN_FILE_PATH}")
        state_dict = torch.load(STM_PRETRAIN_FILE_PATH, weights_only=True, map_location="cpu")  # checkpoints move between CUDA/MPS machines
        agent.model.load_state_dict(state_dict)

    # Utilities for other experiment types
    # Create shared copies of evaluate dataset for individual fine classes 1-5. 1 and 2 grouped together as were pretrained.
    # dataset_options: VALIDATION_DATASET_OPTIONS for the validation sets (held-out training images), else test sets.
    def create_evaluate_individual_fine_class_dataset(exclude_classes_fine_evaluate, dataset_options=None):
        evaluate_dataset = CifarEnv.create_dataset(
            data_file_path = CIFAR_DATA_FILE_PATH,
            mode = Instrumentation.MODE_EVALUATE,
            shared_memory_names = None,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine_evaluate,
            max_instances = None,  # For evaluate, always all
            dataset_options = dataset_options,
        )
        return evaluate_dataset

    def do_evaluate_individual_fine_class(
        epoch:int,
        evaluate_individual_datasets,
        evaluate_individual_dataset_key:str,
        exclude_classes_fine_evaluate,
        results = results_file,
        dataset_options = None,
    ):
        dataset_evaluate = evaluate_individual_datasets[evaluate_individual_dataset_key]
        shared_memory_names_evaluate = dataset_evaluate.get_shared_memory_names()

        CifarEnv.set_dataset_config_for_envs(
            envs = agent.get_envs(Instrumentation.MODE_EVALUATE),
            mode = Instrumentation.MODE_EVALUATE,
            shared_memory_names = shared_memory_names_evaluate,
            exclude_classes_fine = exclude_classes_fine_evaluate,
            max_instances = None,  # All instances
            dataset_options = dataset_options,
        )
        part = "test" if dataset_options is None else "validation"
        accuracy_evaluate = evaluate_current(f"{part}, ['{evaluate_individual_dataset_key}']", epoch)
        results.append_line(
            coarse_classes = str(COARSE_CLASSES),
            fine_classes = str([evaluate_individual_dataset_key]),
            mode = Instrumentation.MODE_EVALUATE,
            epoch = epoch,
            accuracy = accuracy_evaluate,            
        )        

    # Test (and validation) sets per group of fine classes: 1,2 together, as they were pre-trained, then 3, 4, 5.
    exclude_classes_fine_evaluate_by_key = {
        "12": Cifar100Dataset.get_fine_classes([    3,4,5,]),
        "3":  Cifar100Dataset.get_fine_classes([1,2,  4,5,]),
        "4":  Cifar100Dataset.get_fine_classes([1,2,3,  5,]),
        "5":  Cifar100Dataset.get_fine_classes([1,2,3,4,  ]),
    }

    evaluate_individual_datasets = {}
    for key, exclude_classes_fine_evaluate in exclude_classes_fine_evaluate_by_key.items():
        evaluate_individual_datasets[key] = create_evaluate_individual_fine_class_dataset(
            exclude_classes_fine_evaluate = exclude_classes_fine_evaluate,
        )
    logger.info("Eval. individual fine classes using datasets:")
    for key, value in evaluate_individual_datasets.items():
        logger.info(f"Fine classes: {key} --> shared memory: {value.shared_memory_names} size: {len(value)}")

    validation_individual_datasets = {}
    if VAL_HOLDOUT > 0:
        for key, exclude_classes_fine_evaluate in exclude_classes_fine_evaluate_by_key.items():
            validation_individual_datasets[key] = create_evaluate_individual_fine_class_dataset(
                exclude_classes_fine_evaluate = exclude_classes_fine_evaluate,
                dataset_options = VALIDATION_DATASET_OPTIONS,
            )
        for key, value in validation_individual_datasets.items():
            logger.info(f"Validation, fine classes: {key} --> size: {len(value)}")

    def do_evaluate_individual_fine_classes(epoch:int):
        for key, exclude_classes_fine_evaluate in exclude_classes_fine_evaluate_by_key.items():
            logger.info(f"Evaluating @ epoch {epoch} fine classes {key}...")
            do_evaluate_individual_fine_class(
                epoch = epoch,
                evaluate_individual_datasets = evaluate_individual_datasets, 
                evaluate_individual_dataset_key = key,
                exclude_classes_fine_evaluate = exclude_classes_fine_evaluate,
            )
        for key, exclude_classes_fine_evaluate in exclude_classes_fine_evaluate_by_key.items():
            if key not in validation_individual_datasets:
                continue
            logger.info(f"Validating @ epoch {epoch} fine classes {key}...")
            do_evaluate_individual_fine_class(
                epoch = epoch,
                evaluate_individual_datasets = validation_individual_datasets,
                evaluate_individual_dataset_key = key,
                exclude_classes_fine_evaluate = exclude_classes_fine_evaluate,
                results = results_file_val,
                dataset_options = VALIDATION_DATASET_OPTIONS,
            )

    def append_training_line(fine_classes:list[int], epoch:int, accuracy:float):
        # The training line goes into the validation file too, so both files have the head script's layout.
        for results in (results_file, results_file_val):
            if results is None:
                continue
            results.append_line(
                coarse_classes = str(COARSE_CLASSES),
                fine_classes = str(fine_classes),
                mode = Instrumentation.MODE_TRAINING,
                epoch = epoch,
                accuracy = accuracy,
            )

    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_EVALUATE:
        # Evaluate the loaded STM checkpoint on the four test sets once, no training.
        do_evaluate_individual_fine_classes(epoch=0)
        return

    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_FEW_SHOT:

        # Vary fine classes for training.
        # Train: X instances of 3, 4, or 5.
        # Evaluate: All 5, reported separately.
        # Over minibatches to saturation

        exclude_classes_fine_few_shot_training = None
        if FINE_CLASSES == [3]:
            exclude_classes_fine_few_shot_training = Cifar100Dataset.get_fine_classes([1,2, 4,5,])
        elif FINE_CLASSES == [4]:
            exclude_classes_fine_few_shot_training = Cifar100Dataset.get_fine_classes([1,2,3, 5,])
        elif FINE_CLASSES == [5]:
            exclude_classes_fine_few_shot_training = Cifar100Dataset.get_fine_classes([1,2,3,4, ])

        dataset_few_shot_training = CifarEnv.create_dataset(
            data_file_path = CIFAR_DATA_FILE_PATH,
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = None,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine_few_shot_training,
            max_instances = MAX_INSTANCES,
            dataset_options = TRAINING_DATASET_OPTIONS,
        )
        shared_memory_names_few_shot_training = dataset_few_shot_training.get_shared_memory_names()
        logger.info(f"Few-shot training set: {len(dataset_few_shot_training)} images")

        CifarEnv.set_dataset_config_for_envs(
            envs = agent.get_envs(Instrumentation.MODE_TRAINING),
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = shared_memory_names_few_shot_training,
            exclude_classes_fine = exclude_classes_fine_few_shot_training,
            max_instances = MAX_INSTANCES,
            dataset_options = TRAINING_DATASET_OPTIONS,
        )

        for epoch in range(NUM_EPOCHS):
            accuracy_training = do_training(agent = agent, num_steps = TRAINING_STEPS)
            append_training_line(FINE_CLASSES, epoch, accuracy_training)

            is_last_epoch = (epoch == NUM_EPOCHS - 1)
            if EVALUATE_INTERVAL_EPOCHS > 1 and (epoch % EVALUATE_INTERVAL_EPOCHS != 0) and not is_last_epoch:
                continue  # Option to skip evals during pretraining, useful with small batch size

            # Always evaluate on the final epoch, so the last measured point sits at the full
            # training budget rather than at the last multiple of EVALUATE_INTERVAL_EPOCHS.
            do_evaluate_individual_fine_classes(epoch=epoch)

        if args.stm_checkpoint_out is not None:
            os.makedirs(os.path.dirname(os.path.abspath(args.stm_checkpoint_out)), exist_ok=True)
            torch.save(agent.model.state_dict(), args.stm_checkpoint_out)
            logger.info(f"Wrote STM after few-shot training to: {args.stm_checkpoint_out}")

    def do_continual_learning(class_fine:list[int], exclude_classes_fine_continual, start_epoch:int):
        dataset_training_continual = CifarEnv.create_dataset(
            data_file_path = CIFAR_DATA_FILE_PATH,
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = None,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine_continual,
            max_instances = MAX_INSTANCES,
            dataset_options = TRAINING_DATASET_OPTIONS,
        )
        shared_memory_names_continual = dataset_training_continual.get_shared_memory_names()
        logger.info(f"Continual training set, fine class {class_fine}: {len(dataset_training_continual)} images")

        CifarEnv.set_dataset_config_for_envs(
            envs = agent.get_envs(Instrumentation.MODE_TRAINING),
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = shared_memory_names_continual,
            exclude_classes_fine = exclude_classes_fine_continual,
            max_instances = MAX_INSTANCES,  # as dataset_training_continual (None: all); was None, which failed
            dataset_options = TRAINING_DATASET_OPTIONS,
        )

        for epoch in range(NUM_EPOCHS):
            accuracy_training = do_training(agent = agent, num_steps = TRAINING_STEPS)
            append_training_line(class_fine, epoch, accuracy_training)

            do_evaluate_individual_fine_classes(
                epoch = start_epoch + epoch
            )

    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_CONTINUAL:
        continual_exclusions = {
            3: Cifar100Dataset.get_fine_classes([1,2, 4,5,]),
            4: Cifar100Dataset.get_fine_classes([1,2,3, 5,]),
            5: Cifar100Dataset.get_fine_classes([1,2,3,4, ]),
        }

        start_epoch = 0
        for fine_class in FINE_CLASSES:
            logger.info(f"Training fine class {fine_class}")
            exclude_classes_fine_continual = continual_exclusions[fine_class]

            do_continual_learning([fine_class], exclude_classes_fine_continual, start_epoch)
            start_epoch += NUM_EPOCHS

            if args.stm_checkpoint_out is not None:
                phase_path = f"{args.stm_checkpoint_out}_phase{fine_class}.pth"
                os.makedirs(os.path.dirname(os.path.abspath(phase_path)), exist_ok=True)
                torch.save(agent.model.state_dict(), phase_path)
                logger.info(f"Wrote STM after fine class {fine_class} to: {phase_path}")

            steps_complete_training = agent.instrumentation.get_steps(mode = Instrumentation.MODE_TRAINING)
            steps_complete_evaluate = agent.instrumentation.get_steps(mode = Instrumentation.MODE_EVALUATE)    
            logger.info(f"Fine class {fine_class} complete @ {steps_complete_training} (training) / {steps_complete_evaluate} (evaluate)")

if __name__ == "__main__":
    main()
