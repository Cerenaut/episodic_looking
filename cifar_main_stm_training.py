
import logging

import gymnasium as gym
import torch

from environment.cifar.cifar_agent import CifarAgent, CifarAgentConfig
from environment.cifar.cifar_args import CifarArgs
from environment.cifar.cifar_dataset import Cifar100Dataset
from environment.cifar.cifar_env import CifarEnv, CifarEnvConfig
from environment.cifar.cifar_model import CifarModel, CifarModelConfig
from environment.cifar.cifar_results import CifarResults
from util.device import get_device
from util.instrumentation import Instrumentation
from util.log import create_run_path, get_run_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def do_steps_in_mode(agent:CifarAgent, num_steps:int, mode:str):
    CifarEnv.set_mode_for_envs(
        envs = agent.envs,
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

def main():
    args = CifarArgs.parse_args()

    EXPERIMENT_TYPE = args.experiment_type
    BATCH_SIZE = args.batch_size
    SPARSITY = args.sparsity
    MAX_INSTANCES = args.max_instances
    FINE_CLASSES = args.fine_classes
    COARSE_CLASSES = args.coarse_classes
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.epochs
    EVALUATE_INTERVAL_EPOCHS = args.evaluate_epochs
    logger.info(f"Exp.:{EXPERIMENT_TYPE} Fine classes:{FINE_CLASSES} Batch size:{BATCH_SIZE} max. instances:{MAX_INSTANCES} LR: {LEARNING_RATE}")

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

    # The episodic environment doesn't pull from the dataset until exhaustion, because all samples
    # have to be constantly fed with data which persists for multiple steps.
    # Instead, track steps and break periodically for evaluation. We still call these breaks "epochs"
    # although they aren't true epochs in the data.
    TRAINING_STEPS = 4000
    EVALUATE_STEPS = 200 * MAX_EPISODE_STEPS  # Size of test set is 200 images.

    # Construct all the objects needed for the experiment, including model and envs
    max_instances_description = (
        str(MAX_INSTANCES) if MAX_INSTANCES is not None else "500"
    )
    EXPERIMENT_NAME = f"{EXPERIMENT_TYPE}_{FINE_CLASSES}_{max_instances_description}"
    run_root_path = f"cifar_100/{EXPERIMENT_NAME}"
    run_path = get_run_path(
        prefix=run_root_path,
        path="./runs",
    )
    create_run_path(run_path)
    logger.info(f"Run path: {run_path}")

    results_file = CifarResults(run_path = run_path, suffix=EXPERIMENT_TYPE)
    results_file.clear_file()

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

    dataset_training = CifarEnv.create_dataset(
        data_file_path=CIFAR_DATA_FILE_PATH,
        mode=Instrumentation.MODE_TRAINING,
        shared_memory_names=None,
        exclude_classes_coarse=exclude_classes_coarse,
        exclude_classes_fine=exclude_classes_fine,
        max_instances=MAX_INSTANCES,
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
        exclude_classes_fine_training=exclude_classes_fine,
        exclude_classes_fine_evaluate=exclude_classes_fine,
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
        log_path="./runs",
        log_prefix=LOG_PREFIX,
        log_period=LOG_PERIOD,
        # EpisodicAgent
        batch_size=BATCH_SIZE,
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
    )

    agent = CifarAgent(
        config=config,
        model_config=model_config,
    )

    if EXPERIMENT_TYPE == CifarArgs.EXPERIMENT_TYPE_PRETRAIN:

        for epoch in range(NUM_EPOCHS):
            accuracy_training = do_training(agent = agent, num_steps = TRAINING_STEPS)
            results_file.append_line(
                coarse_classes = str(COARSE_CLASSES),
                fine_classes = str(FINE_CLASSES),
                mode = Instrumentation.MODE_TRAINING,
                epoch = epoch,
                accuracy = accuracy_training,            
            )

            if EVALUATE_INTERVAL_EPOCHS > 1 and (epoch % EVALUATE_INTERVAL_EPOCHS != 0):
                continue  # Option to skip evals during pretraining, useful with small batch size

            accuracy_evaluate = do_evaluate(agent = agent, num_steps = EVALUATE_STEPS)
            results_file.append_line(
                coarse_classes = str(COARSE_CLASSES),
                fine_classes = str(FINE_CLASSES),
                mode = Instrumentation.MODE_EVALUATE,
                epoch = epoch,
                accuracy = accuracy_evaluate,            
            )

        # Test save / load STM
        logger.info(f"Writing STM to: {STM_PRETRAIN_FILE_PATH}")
        torch.save(agent.model.state_dict(), STM_PRETRAIN_FILE_PATH)
        return  # End of this experiment type
    else:
        logger.info(f"Reading STM from: {STM_PRETRAIN_FILE_PATH}")
        state_dict = torch.load(STM_PRETRAIN_FILE_PATH, weights_only=True)
        agent.model.load_state_dict(state_dict)

    # Utilities for other experiment types
    # Create shared copies of evaluate dataset for individual fine classes 1-5. 1 and 2 grouped together as were pretrained.
    def create_evaluate_individual_fine_class_dataset(exclude_classes_fine_evaluate):
        evaluate_dataset = CifarEnv.create_dataset(
            data_file_path = CIFAR_DATA_FILE_PATH,
            mode = Instrumentation.MODE_EVALUATE,
            shared_memory_names = None,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine_evaluate,
            max_instances = None,  # For evaluate, always all
        )
        return evaluate_dataset

    def do_evaluate_individual_fine_class(
        epoch:int,
        evaluate_individual_datasets,
        evaluate_individual_dataset_key:str,
        exclude_classes_fine_evaluate,
    ):
        dataset_evaluate = evaluate_individual_datasets[evaluate_individual_dataset_key]
        shared_memory_names_evaluate = dataset_evaluate.get_shared_memory_names()

        CifarEnv.set_dataset_config_for_envs(
            envs = agent.envs,
            mode = Instrumentation.MODE_EVALUATE,
            shared_memory_names = shared_memory_names_evaluate,
            exclude_classes_fine = exclude_classes_fine_evaluate,
            max_instances = None,  # All instances
        )
        accuracy_evaluate = do_evaluate(agent = agent, num_steps = EVALUATE_STEPS)
        results_file.append_line(
            coarse_classes = str(COARSE_CLASSES),
            fine_classes = str([evaluate_individual_dataset_key]),
            mode = Instrumentation.MODE_EVALUATE,
            epoch = epoch,
            accuracy = accuracy_evaluate,            
        )        

    exclude_classes_fine_evaluate_12 = Cifar100Dataset.get_fine_classes([    3,4,5,])
    exclude_classes_fine_evaluate_3  = Cifar100Dataset.get_fine_classes([1,2,  4,5,])
    exclude_classes_fine_evaluate_4  = Cifar100Dataset.get_fine_classes([1,2,3,  5,])
    exclude_classes_fine_evaluate_5  = Cifar100Dataset.get_fine_classes([1,2,3,4,  ])

    evaluate_individual_datasets = {}
    evaluate_individual_datasets["12"] = create_evaluate_individual_fine_class_dataset(
        exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_12,
    )
    evaluate_individual_datasets["3"] = create_evaluate_individual_fine_class_dataset(
        exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_3,
    )
    evaluate_individual_datasets["4"] = create_evaluate_individual_fine_class_dataset(
        exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_4,
    )
    evaluate_individual_datasets["5"] = create_evaluate_individual_fine_class_dataset(
        exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_5,
    )
    logger.info("Eval. individual fine classes using datasets:")
    for key, value in evaluate_individual_datasets.items():
        logger.info(f"Fine classes: {key} --> shared memory: {value.shared_memory_names} size: {len(value)}")

    def do_evaluate_individual_fine_classes(epoch:int):
        logger.info(f"Evaluating @ epoch {epoch} fine classes 1,2...")
        do_evaluate_individual_fine_class(
            epoch = epoch,
            evaluate_individual_datasets = evaluate_individual_datasets, 
            evaluate_individual_dataset_key = "12",
            exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_12,
        )    
        logger.info(f"Evaluating @ epoch {epoch} fine classes 3...")
        do_evaluate_individual_fine_class(
            epoch = epoch,
            evaluate_individual_datasets = evaluate_individual_datasets, 
            evaluate_individual_dataset_key = "3",
            exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_3,
        )
        logger.info(f"Evaluating @ epoch {epoch} fine classes 4...")
        do_evaluate_individual_fine_class(
            epoch = epoch,
            evaluate_individual_datasets = evaluate_individual_datasets, 
            evaluate_individual_dataset_key = "4",
            exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_4,
        )    
        logger.info(f"Evaluating @ epoch {epoch} fine classes 5...")
        do_evaluate_individual_fine_class(
            epoch = epoch,
            evaluate_individual_datasets = evaluate_individual_datasets, 
            evaluate_individual_dataset_key = "5",
            exclude_classes_fine_evaluate = exclude_classes_fine_evaluate_5,
        )    

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
        )
        shared_memory_names_few_shot_training = dataset_few_shot_training.get_shared_memory_names()

        CifarEnv.set_dataset_config_for_envs(
            envs = agent.envs,
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = shared_memory_names_few_shot_training,
            exclude_classes_fine = exclude_classes_fine_few_shot_training,
            max_instances = MAX_INSTANCES,
        )

        for epoch in range(NUM_EPOCHS):
            accuracy_training = do_training(agent = agent, num_steps = TRAINING_STEPS)
            results_file.append_line(
                coarse_classes = str(COARSE_CLASSES),
                fine_classes = str(FINE_CLASSES),
                mode = Instrumentation.MODE_TRAINING,
                epoch = epoch,
                accuracy = accuracy_training,            
            )

            if EVALUATE_INTERVAL_EPOCHS > 1 and (epoch % EVALUATE_INTERVAL_EPOCHS != 0):
                continue  # Option to skip evals during pretraining, useful with small batch size

            do_evaluate_individual_fine_classes(epoch=epoch)

    def do_continual_learning(class_fine:list[int], exclude_classes_fine_continual, start_epoch:int):
        dataset_training_continual = CifarEnv.create_dataset(
            data_file_path = CIFAR_DATA_FILE_PATH,
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = None,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine_continual,
            max_instances = MAX_INSTANCES,
        )
        shared_memory_names_continual = dataset_training_continual.get_shared_memory_names()

        CifarEnv.set_dataset_config_for_envs(
            envs = agent.envs,
            mode = Instrumentation.MODE_TRAINING,
            shared_memory_names = shared_memory_names_continual,
            exclude_classes_fine = exclude_classes_fine_continual,
            max_instances = None,  # All instances
        )

        for epoch in range(NUM_EPOCHS):
            accuracy_training = do_training(agent = agent, num_steps = TRAINING_STEPS)
            results_file.append_line(
                coarse_classes = str(COARSE_CLASSES),
                fine_classes = str(class_fine),
                mode = Instrumentation.MODE_TRAINING,
                epoch = epoch,
                accuracy = accuracy_training,            
            )

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

            steps_complete_training = agent.instrumentation.get_steps(mode = Instrumentation.MODE_TRAINING)
            steps_complete_evaluate = agent.instrumentation.get_steps(mode = Instrumentation.MODE_EVALUATE)    
            logger.info(f"Fine class {fine_class} complete @ {steps_complete_training} (training) / {steps_complete_evaluate} (evaluate)")

if __name__ == "__main__":
    main()
