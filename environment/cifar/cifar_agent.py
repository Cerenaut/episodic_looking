import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from agent.episodic_agent import EpisodicAgent, EpisodicAgentConfig

from environment.cifar.cifar_env import CifarEnv
from environment.cifar.cifar_model import CifarModel, CifarModelConfig
from environment.cifar.cifar_results import CifarResults
from util.log import loss_to_float_with_norm, tensor_to_float_with_norm
from util.log_writer import ScalarLogWriter
from util.optimizer import ModelOptimizer, ModelOptimizerConfig
from util.reinforcement_learning.policy_util import PolicyUtil

logger = logging.getLogger(__name__)


@dataclass
class CifarAgentConfig(EpisodicAgentConfig):
    image_shape: list[int] = field(default_factory=list) 
    learning_rate:float = 0.1
    momentum:float = 0.0

class CifarAgent(EpisodicAgent):
    """
    Agent for the Cifar-100 dataset.
    """

    def __init__(
        self, 
        config:CifarAgentConfig,
        model_config:CifarModelConfig,
        device = None,
    ):
        self.model_config = model_config
        super().__init__(config, device)

    def model_create(self):
        self.model = CifarModel(
            config = self.model_config,
            device = self.device,
        )

        self.create_optimizers()

        self.reward_previous_default = self.model.get_class_reward_default(self.config.batch_size)
        self.previous_policy_data = None
        self.current_policy_data = None

    def create_optimizers(self):
        optimizer_config = self.create_optimizer_config(
            learning_rate = self.config.learning_rate,
            momentum = self.config.momentum,
        )
        parameters = self.model.get_trainable_parameters()
        self.optimizer = ModelOptimizer(
            config = optimizer_config,
            parameters = parameters,
        )

    def create_optimizer_config(self, learning_rate:float, momentum:float):
        optimizer_config = ModelOptimizerConfig(
            name = "Optimizer",
            optimizer_type = ModelOptimizer.OPTIMIZER_SGD,
            learning_rate = learning_rate,
            momentum = momentum,
            weight_decay = 0,
            clip_grad_norm = 0,
        )
        return optimizer_config

    def optimize(self, loss, optimize:bool = True):
        parameters = self.model.get_trainable_parameters()
        if optimize:
            self.optimizer.optimize(
                loss = loss,
                parameters = parameters,
            )
        else:
            self.optimizer.clear_gradients()  # don't want them accumulating

    def model_actions(self) -> np.ndarray:
        self.output_actor = self.model.do_actor(
            input = self.state.obs_1_history_tensor,
        )
        self.set_bias(
            self.output_actor.sample.detach().clone()
        )
        return None  # no actions to env; only perceptual

    def copy_obs_final(
            self,
            obs,  # dict
            final_obs,  # numpy array of dict
            batch_index:int,
    ):
        final_obs_sample = final_obs[batch_index]
        obs[CifarEnv.OBSERVATION_KEY_CLASS][batch_index] = final_obs_sample[CifarEnv.OBSERVATION_KEY_CLASS]
        obs[CifarEnv.OBSERVATION_KEY_IMAGE][batch_index,:] = final_obs_sample[CifarEnv.OBSERVATION_KEY_IMAGE]

    def model_update(self):

        # do class model: Tries to classify, and produces rewards for the bias.
        class_labels = self.observation_to_class(self.state.obs_2_final)  # work on ending episode, if any
        self.class_distribution_targets = self.model.get_class_distribution_targets(
            class_labels = class_labels
        )
        self.class_distribution_predicted_logits = self.classifier_logits  # with grads; was produced with observation_to_tensor(obs_hist_2)
        class_logits_detached = self.class_distribution_predicted_logits.detach()

        self.class_distribution_predicted = F.softmax(class_logits_detached, dim=1)
        self.class_distribution_rewards = self.model.get_class_reward(
            class_distribution_predicted_logits = class_logits_detached,       
            class_distribution_predicted = self.class_distribution_predicted,
            class_distribution_targets = self.class_distribution_targets.detach(),
            reward_previous = self.reward_previous,
        )  # rewards for bias this step

        # Do A2C loss. First, critic
        self.advantage = self.model.get_advantage(
            input_1 = self.state.obs_1_history_tensor,
            rewards = self.class_distribution_rewards,
            terminated = self.state.terminated,
            input_2 = self.state.obs_2_history_tensor,
        )
        self.model.update_critic_loss(self.advantage)  # un-normalized advantage

        # Actor
        self.advantage_normalized = self.model.get_advantage_normalized(self.advantage.detach())

        self.previous_policy_data = self.current_policy_data
        self.current_policy_data = PolicyUtil.create_policy_data(
            policy_model_input = self.state.obs_1_history_tensor,
            policy_model_output = self.output_actor,
            advantage = self.advantage_normalized,
        )

        self.model.update_actor_loss(
            previous_policy_data = self.previous_policy_data,
            policy_output = self.output_actor,
            advantage = self.advantage_normalized,
        )

        # Optimize
        optimize = self.instrumentation.is_mode_training()
        self.optimize(
            loss = self.model.get_total_loss(),
            optimize = optimize,
        )

    def reset(self):
        super().reset()
        self.previous_policy_data = None
        self.current_policy_data = None
        self.reset_reward_previous()
        self.reset_bias()

        #self.reset_intra_episode_metrics()

    def state_update(self, reset_mask:torch.Tensor):
        super().state_update(reset_mask)

        # Clear any old state on episode reset
        self.update_reward_previous(self.class_distribution_rewards)
        self.reset_reward_previous(reset_mask)  # reset obs_2 for complete episodes 
        self.reset_bias(reset_mask)

    def reset_bias(self, reset_mask:torch.Tensor|None = None):
        if reset_mask is None:
            # Since gating = 2*sigmoid(gain*bias) and gain = 1, then no change is a bias of 0.
            bias_size = self.model.get_bias_size()
            self.bias = torch.zeros(
                (
                    self.config.batch_size,
                    bias_size,
                ),
                device = self.device,
            )
        else:
            self.bias[reset_mask] = 0.0

    def get_bias(self) -> torch.Tensor:
        return self.bias

    def set_bias(self, bias:torch.Tensor):
        self.bias = bias

    def reset_reward_previous(self, reset_mask:torch.Tensor|None = None):
        if reset_mask is None:
            self.reward_previous = self.reward_previous_default.clone()
        else:
            self.reward_previous[reset_mask] = self.reward_previous_default[reset_mask]

    def update_reward_previous(self, reward:torch.Tensor):
        self.reward_previous = reward

    def actions_to_tensor(self, actions: np.ndarray|None) -> torch.Tensor:
        if actions is not None:
            actions = actions["actions"]  # extract array from dict
        return super().actions_to_tensor(actions)

    def observation_to_tensor(self, observation) -> torch.Tensor:
        """
        Observations are filtered by the current bias when they occur.
        """
        with torch.no_grad():
            key = CifarEnv.OBSERVATION_KEY_IMAGE
            image_array = observation[key]
            bias_detached = self.get_bias().detach()  # current bias
            image_tensor = torch.from_numpy(image_array).to(self.device)
            logits, encoding = self.model.do_classifier(
                image = image_tensor,
                bias = bias_detached,
            )
            logits_detached = logits.detach()
            observation_and_bias = torch.cat(
                [
                    encoding.detach(),  # ensure no grads
                    bias_detached,  # bias which was applied
                    logits_detached,  # resulting logits
                ], 
                dim=1,
            )
            self.classifier_logits = logits_detached
            return observation_and_bias
 
    def observation_to_class(self, observation) -> torch.Tensor:
        key = CifarEnv.OBSERVATION_KEY_CLASS
        image_classes = observation[key]
        t_image_classes = torch.tensor(image_classes, dtype=torch.long, device=self.device)
        return t_image_classes

    def add_log_values(self, log_writer:ScalarLogWriter):
        super().add_log_values(log_writer)

        target_indices = self.class_distribution_targets
        max_predicted_indices = torch.argmax(self.class_distribution_predicted, dim=1)
        is_correct = (max_predicted_indices == target_indices).long()
        batch_indices = torch.arange(self.config.batch_size)
        predicted_target_probabilities = self.class_distribution_predicted[batch_indices, target_indices]

        log_writer.add_scalar("freq_max_correct", tensor_to_float_with_norm(is_correct, self.config.batch_size))
        log_writer.add_scalar("p_target_class", tensor_to_float_with_norm(predicted_target_probabilities, self.config.batch_size))

        mask_terminated = self.state.terminated != 0
        mask_truncated = self.state.truncated != 0
        end_episode_mask = mask_terminated | mask_truncated

        is_correct_end_episode = is_correct[end_episode_mask]
        prediction_end_episode = predicted_target_probabilities[end_episode_mask]       
        num_end_episode_samples = is_correct_end_episode.shape[0]
        if num_end_episode_samples > 0:
            log_writer.add_scalar("freq_max_correct_end", tensor_to_float_with_norm(is_correct_end_episode, num_end_episode_samples))
            log_writer.add_scalar("p_target_class_end", tensor_to_float_with_norm(prediction_end_episode, num_end_episode_samples))

            self.cumulative_correct += is_correct_end_episode.sum()
            self.cumulative_samples += num_end_episode_samples

        log_writer.add_scalar("reward", tensor_to_float_with_norm(self.state.rewards, self.config.batch_size))  # sum / batch_size
        log_writer.add_scalar("advantage-normalized", tensor_to_float_with_norm(self.advantage_normalized, self.config.batch_size))
        log_writer.add_scalar("bias-sum", tensor_to_float_with_norm(self.bias.sum(dim=1), self.config.batch_size))
        log_writer.add_scalar("loss-actor", loss_to_float_with_norm(self.model.loss_actor_scaled, 1))
        log_writer.add_scalar("loss-critic", loss_to_float_with_norm(self.model.loss_critic_scaled, 1))

        #self.update_intra_episode_metrics()

    def reset_intra_episode_metrics(self):
        self.episode_step = 0
        self.results = CifarResults(run_path="./", suffix=f"intra_episode_{self.instrumentation.get_mode()}")  # reuse this
        self.results.clear_file()

    def update_intra_episode_metrics(self):
        target_indices = self.class_distribution_targets
        max_predicted_indices = torch.argmax(self.class_distribution_predicted, dim=1)
        freq_correct = (max_predicted_indices == target_indices).long().sum().item() / self.config.batch_size
        bias_sum = self.bias.abs().sum().item() / self.config.batch_size
        mean_reward = self.state.rewards.mean()
        log_probs = F.log_softmax(self.class_distribution_predicted_logits, dim=1)
        entropy = -(self.class_distribution_predicted * log_probs).sum().item() / self.config.batch_size
        metrics = f"{self.episode_step},{freq_correct},{bias_sum},{mean_reward},{entropy}\n"
        self.results.append_file(text=metrics)

        #print(f"Eps. step:{self.episode_step}, Freq. correct:{freq_correct} Bias abs. sum:{bias_sum} mean reward:{mean_reward} entropy: {entropy}")

        self.episode_step += 1
        if self.state.completed[0]:  # all sync
            self.episode_step = 0

    def reset_cumulative_accuracy(self):
        self.cumulative_correct = 0
        self.cumulative_samples = 0

    def get_cumulative_accuracy(self):
        if self.cumulative_samples > 0:
            accuracy = self.cumulative_correct / self.cumulative_samples
        else:
            accuracy = 0
        logger.info(f"{self.cumulative_correct} / {self.cumulative_samples} = Accuracy: {accuracy}")
        return accuracy