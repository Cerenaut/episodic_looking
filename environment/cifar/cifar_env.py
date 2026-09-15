from __future__ import annotations

import logging
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from environment.cifar.cifar_dataset import Cifar100Dataset, CifarSharedMemoryNames
from util.instrumentation import Instrumentation

logger = logging.getLogger(__name__)

@dataclass
class CifarEnvConfig:

    data_file_path:str
    image_shape: list[int] = field(default_factory=list) 
    num_classes:int = 0
    max_steps:int = 0  # in episode
    shared_memory_names_training:CifarSharedMemoryNames|None = None
    shared_memory_names_evaluate:CifarSharedMemoryNames|None = None
    exclude_classes_coarse:set[int]|None = field(default_factory=None)
    exclude_classes_fine_training:set[int]|None = field(default_factory=None)
    exclude_classes_fine_evaluate:set[int]|None = field(default_factory=None)
    max_instances_training:int|None = None
    max_instances_evaluate:int|None = None


class CifarEnv(gym.Env):

    OBSERVATION_KEY_IMAGE = "image"
    OBSERVATION_KEY_CLASS = "class"

    """
    Presents classification of a Cifar 100 image over an episode as a RL problem.
    The model is rewarded for correct classification.

    Data from:
    https://cave.cs.toronto.edu/kriz/cifar.html
    """

    def __init__(
        self,
        config:CifarEnvConfig,
    ):
        # Env basic state
        self.steps = 0
        self.episode_terminated = False
        self.episode_truncated = False
        self.config = config

        self.dataset = None

        # Mode & image
        self.mode = None  # Set on reset()
        self.image_index = None  # Set on reset()
        self.state_dict = None  # Set on reset()
        self.obs_cache = None

        observation_space_dict = {}
        self._add_observation_spaces(observation_space_dict)
        self.observation_space = gym.spaces.Dict(observation_space_dict)

        # Actions
        action_space_dict = {}
        self._add_action_spaces(action_space_dict)
        self.action_space = gym.spaces.Dict(action_space_dict)

    def get_mode(self) -> str:
        return self.mode

    def set_mode(self, mode:str):
        if self.mode == mode:
            return
        self.mode = mode
        self._create_dataset()

    @staticmethod
    def set_dataset_config_for_envs(
            envs, 
            mode:str,
            shared_memory_names:CifarSharedMemoryNames,
            exclude_classes_fine:set[int],
            max_instances:int|None,
    ):
        envs.call(
            "set_dataset_config",
            mode=mode,
            shared_memory_names=shared_memory_names,
            exclude_classes_fine=exclude_classes_fine,
            max_instances=max_instances,
        )

    def set_dataset_config(
        self,
        mode:str,
        shared_memory_names:CifarSharedMemoryNames,
        exclude_classes_fine:set[int],
        max_instances:int|None,
    ):
        """
        Allows all the dataset filtering options we want to vary during training to be varied in combination.
        Make sure shared_memory_name refers to a dataset which matches this configuration or bad things will happen.
        """
        if mode == Instrumentation.MODE_EVALUATE:
            self.config.shared_memory_names_evaluate = shared_memory_names
            self.config.exclude_classes_fine_evaluate = exclude_classes_fine
            self.config.max_instances_evaluate = max_instances

        elif mode == Instrumentation.MODE_TRAINING:
            self.config.shared_memory_names_training = shared_memory_names
            self.config.exclude_classes_fine_training = exclude_classes_fine
            self.config.max_instances_training = max_instances

        else:
            raise ValueError(f"Mode: {mode} not recognized.")
        
        self.mode = mode
        self._create_dataset()
        
    def _create_dataset(self):
        self.image_index = None

        shared_memory_names = self.config.shared_memory_names_training
        exclude_classes_fine = self.config.exclude_classes_fine_training
        max_instances = self.config.max_instances_training
        
        if self.mode == Instrumentation.MODE_EVALUATE:
            shared_memory_names = self.config.shared_memory_names_evaluate
            exclude_classes_fine = self.config.exclude_classes_fine_evaluate
            max_instances = self.config.max_instances_evaluate

        #logger.info(f"create_dataset(): shared mem.: {shared_memory_name} max. instances: {max_instances}")
        self.dataset = CifarEnv.create_dataset(
            data_file_path = self.config.data_file_path,
            mode = self.mode,
            shared_memory_names = shared_memory_names,
            exclude_classes_coarse = self.config.exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine,
            max_instances = max_instances,
        )

    @staticmethod
    def set_mode_for_envs(envs, mode:str):
        envs.call(
            "set_mode",
            mode=mode,
        )

    @staticmethod
    def create_dataset(
        data_file_path:str, 
        mode:str, 
        shared_memory_names:CifarSharedMemoryNames|None,
        exclude_classes_coarse:set[int]|None,
        exclude_classes_fine:set[int]|None,
        max_instances:int|None,
        as_tensor:bool = False,
    ) -> Cifar100Dataset:
        # Read data file
        #logger.info(f"Loading data file: '{data_file_path}' mode:{mode} (shared mem:{shared_memory_names})...")
        is_training = False
        if mode == Instrumentation.MODE_TRAINING:
            is_training = True
        dataset = Cifar100Dataset(
            file_path = data_file_path,
            label_type = Cifar100Dataset.LABEL_TYPE_COARSE,
            training = is_training,
            exclude_classes_coarse = exclude_classes_coarse,
            exclude_classes_fine = exclude_classes_fine,
            max_instances = max_instances,
            shared_memory_names = shared_memory_names,
            as_tensor = as_tensor,  # keep as numpy
        )
        return dataset

    @staticmethod
    def set_image_for_envs(
            envs, 
            mode:str,
            image_index:int,
    ):
        envs.call(
            "set_image",
            mode=mode,
            image_index=image_index,
        )

    def get_num_images(self) -> int:
        return len(self.dataset)

    def set_image(self,
        mode:str, 
        image_index:int,
    ):
        """
        Explicitly set a specific pair of images from the specified modes.
        """
        self.set_mode(mode)
        self.image_index = image_index

    def set_random_image(self):
        # Pick an image randomly. 
        num_images = self.get_num_images()
        self.image_index = np.random.randint(0, num_images)
    
    def get_state_dict(self) -> dict:
        # Lazy creation; remains valid until end of episode.
        if self.state_dict is None:
            image, label = self.dataset[self.image_index]
            self.state_dict = {
                "image": image, #.numpy(),
                "class": label,
            } 
        return self.state_dict

    def _add_observation_spaces(self, observation_space_dict:dict):
        observation_space_dict[CifarEnv.OBSERVATION_KEY_IMAGE] = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.config.image_shape,
            dtype=np.float32,
        )    
        observation_space_dict[CifarEnv.OBSERVATION_KEY_CLASS] = gym.spaces.Discrete(
            self.config.num_classes
        )
        logger.info(f"observation_space:{observation_space_dict}")

    def _add_action_spaces(self, action_space_dict:dict):
        """
        Action space is 2 values - match, or no match.
                
        :param self: Description
        :param action_space_dict: Description
        :type action_space_dict: dict
        """
        action_space_dict["actions"] = gym.spaces.Discrete(self.config.num_classes)
        logger.info(f"action_space:{action_space_dict}")

    def _get_info(self):
        """
        Compute auxiliary information for debugging.

        Returns:
            dict: Info about state of environment.
        """
        state_dict = self.get_state_dict()

        return {
            "step": self.steps,
            "image":{
                "index": self.image_index,
                "class": state_dict["class"],
            },
        }

    def _get_obs(self) -> dict:
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """

        # Lazy creation; remains valid until end of episode.
        if self.obs_cache is None:
        
            state_dict = self.get_state_dict()
            image = state_dict["image"] #.squeeze(axis=0)  # [1,28,28] --> [28x28]
            image_class = state_dict["class"]

            self.obs_cache = {
                CifarEnv.OBSERVATION_KEY_IMAGE: image,
                CifarEnv.OBSERVATION_KEY_CLASS: image_class,
            }

        return self.obs_cache
    
    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.steps = 0
        self.episode_terminated = False  # task achieved, goal reached, ended, etc.
        self.episode_truncated = False  # ran out of time
        self._reset_state()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _reset_state(self):
        self.reward = 0.0
        self.set_random_image()
        self.state_dict = None
        self.obs_cache = None  # reset cached obs

    def _update_state(self, action):
        state_dict = self.get_state_dict()
        image_class = state_dict["class"]
        action_class = int(action["actions"])

        if action_class == image_class:
            self.reward = 1.0
        else:
            self.reward = 0.0

    def _get_reward(self) -> float:
        return self.reward

    def step(self, action):
        """
        Execute one timestep within the environment.
        On complete, return final obs of ending episode.
        Vector wrapper will call reset() and replace the final obs.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        # Detect terminated or truncated previous update, which would have emitted a reward
        # If complete, stop updating state.
        episode_complete = self.is_episode_complete()
        #print(f"Env.step() complete={episode_complete}")
        if not episode_complete:
            self._update_state(action)

        # Only one of truncated and terminated should be true.
        # Check and latch if episode completion criteria reached
        if not self.episode_truncated:
            self.episode_terminated = self.episode_terminated or self._is_episode_terminated()

        # Check and latch if episode timeout reached
        if not self.episode_terminated:
            self.episode_truncated = self.episode_truncated or self._is_episode_truncated()

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = self._get_reward()
        observation = self._get_obs()  # will immediately display completion observation
        info = self._get_info()

        # if not episode_complete:
        self.steps += 1
        return observation, reward, self.episode_terminated, self.episode_truncated, info

    def is_episode_complete(self) -> bool:
        return self.episode_terminated or self.episode_truncated

    def _is_episode_truncated(self) -> bool:
        """
        Specify logic to determine end of episode criteria.
        Otherwise, will end on timeout. 
        """
        # e.g. max_steps=10, then steps >= 9 is truncated
        if self.steps >= (self.config.max_steps -1):
            return True
        return False
    
    def _is_episode_terminated(self) -> bool:
        return False  # Always truncated
