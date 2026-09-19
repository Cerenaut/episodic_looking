import logging
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from util.device import get_device
from util.instrumentation import Instrumentation, InstrumentationConfig
from util.log import tensor_to_float_with_norm
from util.log_writer import ScalarLogWriter
from util.observation_history import ObservationHistory, ObservationHistoryConfig

logger = logging.getLogger(__name__)

@dataclass
class AgentState:

    actions_before:np.ndarray = None  # as passed to env

    obs_1:Any = None
    obs_1_tensor:torch.Tensor = None
    obs_1_history_tensor:torch.Tensor = None  # Last n observations
    info_1:dict[str, Any] = None

    actions:np.ndarray = None  # as passed to env

    obs_2:Any = None  # NON-final ie continuing and first obs. of new episodes.
    obs_2_final:Any = None  # contains last obs of ending episodes
    obs_2_tensor:torch.Tensor = None  # contains final obs of ending episodes
    obs_2_history_tensor:torch.Tensor = None  # Last n observations up to and including final obs for completed episodes
    info_2:dict[str, Any] = None

    terminated:torch.Tensor = None
    truncated:torch.Tensor = None
    completed:torch.Tensor = None
    rewards:torch.Tensor = None


@dataclass
class EpisodicAgentConfig(InstrumentationConfig):
    batch_size: int = 0
    history_size: int = 0
    action_size: int = 0
    observation_size: int = 0
    random_policy: bool = False
    environment_id: str = None
    async_env:bool = False
    max_episode_steps:int = 0

class EpisodicAgent:
    """
    Base class for agent used in multiple environments. Manages context buffer of observations.
    Has a lot of utility functionality (e.g. logging) and defines how each episode step is executed.
    """

    def __init__(
        self, 
        config:EpisodicAgentConfig,
        device = None,
    ):
        self.device = device
        if self.device is None:
            self.device = get_device()

        self.config = config
        self.state = None

        observation_history_config = ObservationHistoryConfig(
            batch_size= self.config.batch_size,
            history_size= self.config.history_size,
            observation_size = self.config.observation_size,
        )
        self.observation_history = ObservationHistory(
            config = observation_history_config,
            device = self.device,
        )

        self.instrumentation = Instrumentation(self.config)
        self.instrumentation.write_config(self.config)

        logger.info("Creating environments...")
        self.envs = self.create_environments()

        logger.info("Creating models...")
        self.model_create()

        logger.info("Agent init complete.")

    def create_environments(self) -> gym.vector.VectorEnv:
        # Agent step() assumes https://farama.org/Vector-Autoreset-Mode SAME_STEP
        # i.e. resets in same step, final obs available in infos.
        max_episode_steps = self.config.max_episode_steps
        environment_id = self.config.environment_id
        def make_env():
            env = gym.make(
                environment_id,
            )

            if max_episode_steps > 0:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        if self.config.async_env:
            envs = gym.vector.AsyncVectorEnv(
                [make_env for _ in range(self.config.batch_size)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,  # Force legacy behavior to allow every step to train without mask
            )
        else:
            envs = gym.vector.SyncVectorEnv(
                [make_env for _ in range(self.config.batch_size)],
                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,  # Force legacy behavior to allow every step to train without mask
            )
        return envs

    def set_mode(self, mode:str):
        self.instrumentation.set_mode(mode)
        self.reset()  # inc. reset envs

    def reset(self):
        # Envs
        obs_1, info_1 = self.envs.reset()

        # State
        self.state = AgentState(
            obs_1 = obs_1,
            info_1 = info_1,
        )

        # History
        self.observation_history.reset()

    def state_update(self, reset_mask:torch.Tensor):
        # Don't need to reset the state, as it's always only the latest obs.
        # Do need to reset the history to clear out old episodes.
        self.state = AgentState(
            obs_1 = self.state.obs_2,  # NOT final - the initial state of new episode
            info_1 = self.state.info_2,
            actions_before = self.state.actions,
        )
        self.observation_history.reset(reset_mask=reset_mask)

    def do_steps(self, num_steps:int):
        for step in range(num_steps):
            self.step(num_steps)

    def step(self, num_steps:int = 0) -> int:
        # Convert the most recent observation into a tensor (somehow) and add it to the context
        self.state.obs_1_tensor = self.observation_to_tensor(
            observation=self.state.obs_1,
        )
        self.observation_history.update(observation=self.state.obs_1_tensor)
        self.state.obs_1_history_tensor = self.observation_history_to_tensor(self.observation_history)

        self.state.actions_1_tensor = self.actions_to_tensor(            
            actions = self.state.actions_before,
        )

        self.state.actions = self.model_actions()
        if self.config.random_policy or self.state.actions is None:
            self.state.actions = self.envs.action_space.sample()
        
        self.env_update()
        self.get_obs_final()
        self.state.obs_2_tensor = self.observation_to_tensor(
            observation=self.state.obs_2_final,
        )
        observation_history_final = self.observation_history.clone()
        observation_history_final.update(observation=self.state.obs_2_tensor)
        self.state.obs_2_history_tensor = self.observation_history_to_tensor(observation_history_final)

        self.state.actions_2_tensor = self.actions_to_tensor(
            actions=self.state.actions,
        )

        self.model_update()  # uses self.state.obs_2_history_tensor

        # Logging etc
        global_step = self.update_instrumentation()
        self.instrumentation.print_steps(num_steps = num_steps)

        self.state_update(reset_mask=self.state.completed)
        return global_step

    def observation_to_tensor(self, observation) -> torch.Tensor:
        """
        Do whatever preprocessing / unpacking needed to transform the env observation
        into a torch.Tensor suitable as model input. 
        If you want to observe the actions, concat them too.
        actions may be None if first step of episode - substitute with something appropriate.
        """
        return torch.tensor(observation, device=self.device)

    def actions_one_hot(self, actions: np.ndarray) -> torch.Tensor:
        actions_tensor = torch.tensor(actions, device=self.device)
        actions_one_hot = torch.nn.functional.one_hot(
            actions_tensor, 
            num_classes=self.config.action_size
        ).float()
        return actions_one_hot

    def actions_to_tensor(self, actions: np.ndarray|None) -> torch.Tensor:
        if actions is None:
            return torch.zeros(
                (self.config.batch_size, self.config.action_size), 
                dtype=torch.float,
                device=self.device, 
            )

        actions_one_hot = self.actions_one_hot(actions)
        return actions_one_hot

    def actions_one_hot_to_action_indices(self, action_mask: torch.Tensor):
        """
        Converts one-hot tensor to integer indices for env.step()
        
        Args:
            actions_mask: [B, A] one-hot tensor
        Returns:
            indices: [B] integer tensor
        """
        # Since it's one-hot, argmax gives us the index of the '1'
        return torch.argmax(action_mask, dim=-1)

    def observation_history_to_tensor(self, observation_history:ObservationHistory) -> torch.Tensor:
        return observation_history.get_tensor_vector()

    def action_history_to_tensor(self, action_history:ObservationHistory) -> torch.Tensor:
        return action_history.get_tensor_vector()

    def env_update(self):
        obs_2, rewards, terminated, truncated, infos = self.envs.step(self.state.actions)
        self.state.obs_2 = obs_2
        self.state.rewards = torch.tensor(rewards).float().to(self.device)  # cast before .to(): MPS has no float64
        self.state.info_2 = infos
        self.state.terminated = torch.from_numpy(terminated).to(self.device).float()
        self.state.truncated = torch.from_numpy(truncated).to(self.device).float()
        self.state.completed = terminated | truncated

    def get_obs_final(self):
        self.state.obs_2_final = self.state.obs_2.copy()

        self.episode_reward_mean = None
        self.episode_length_mean = None

        if any(self.state.completed):
            batch_size = self.state.obs_1_tensor.shape[0]

            # Check if the keys exist in the batch at all
            final_obs_key = "final_obs"
            final_info_key = "final_info"

            # These will be arrays of length [batch_size]
            final_obs_batch = self.state.info_2.get(final_obs_key)
            final_info_batch = self.state.info_2.get(final_info_key)

            episode_rewards = []
            episode_lengths = []

            for b in range(batch_size):
                if self.state.completed[b]:
                    self.copy_obs_final(
                        obs=self.state.obs_2_final, 
                        final_obs=final_obs_batch, 
                        batch_index=b,
                    )
                    episode_reward = final_info_batch["episode"]["r"][b]
                    episode_length = final_info_batch["episode"]["l"][b]
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

            # TensorBoard Episodic Logging
            if episode_rewards:  # if is not empty
                self.episode_reward_mean = np.mean(episode_rewards)
                self.episode_length_mean = np.mean(episode_lengths)

    def copy_obs_final(
            self,
            obs,  # obs type eg numpy array or dict
            final_obs,  # numpy array of obs type
            batch_index:int,
    ):
        """
        Copy the final observation of an ending episode of batch sample batch_index 
        from final_obs into the obs.    
        :param obs: Description
        :param final_obs: Description
        :param batch_index: Description
        :type batch_index: int
        """
        obs[batch_index] = final_obs[batch_index]

    def get_num_actions(self) -> int:
        num_actions = self.envs.single_action_space.n
        return num_actions

    def model_create(self):
        pass

    def model_actions(self) -> np.ndarray:
        return None

    def model_update(self):
        pass

    def update_instrumentation(self):
        global_step = self.instrumentation.increment_steps()
        log_writer = self.instrumentation.get_log_writer()
        self.add_log_values(log_writer)
        self.instrumentation.flush_log_writers()
        return global_step

    def add_log_values(self, log_writer:ScalarLogWriter):
        batch_size = self.state.rewards.shape[0]
        log_writer.add_scalar("reward", tensor_to_float_with_norm(self.state.rewards, batch_size))  # sum / batch_size

        if self.episode_reward_mean is not None:
            log_writer.add_scalar("episode-reward", self.episode_reward_mean)
        if self.episode_length_mean is not None:
            log_writer.add_scalar("episode-length", self.episode_length_mean)
