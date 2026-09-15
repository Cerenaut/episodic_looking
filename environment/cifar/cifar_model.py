import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from environment.cifar.cifar_classifier import CifarClassifier
from model.dense import DenseModel, DenseModelConfig
from model.resnet import ResNetConfig
from model.sparse.sparse_dense_model import SparseActivationDenseModelConfig
from model.sparse.sparse_distributed_model import SparseDistributedModel
from util.reinforcement_learning.policy_util import (
    PolicyConfig,
    PolicyData,
    PolicyModelOutput,
    PolicyUtil,
)
from util.reinforcement_learning.rl_util import (
    normalize_advantage,
)

logger = logging.getLogger(__name__)

@dataclass
class CifarModelConfig:

    encoder_ensemble_size:int = 1
    encoder_sparsity:int = 32

    history_size:int = 1

    model_hidden_size:int = 1000
    model_hidden_size_factor:int = 1
    model_nonlinearity:str = "leaky-relu"
    model_layers:int = 2

    discount_factor:float = 0.9
    normalize_advantage:bool = True
    normalize_advantage_epsilon: float = 0.0001
    normalize_advantage_clamp:float = None

    reward_scale: float = 1.0
    reward_type:str = ""

    loss_actor_scale: float = 1.0
    loss_critic_scale: float = 1.0
    loss_entropy_scale: float = 1.0
    loss_class_scale: float = 1.0

    loss_huber_delta: float = 1.0
    loss_actor_type: str = ""
    loss_critic_type: str = ""

    num_classes: int = 0
    classifier_model_file:str = ""
    classifier_bias_stage:int = 1
    policy_std:float|None = None
    

class CifarModel:
    """
    Uses SparseDistributedModels to learn value and policy models.
    Accepts a moving context window of recent observations as input.
    The context window is assumed to be constant size with zeros for empty observations.
    """

    REWARD_TYPE_ENTROPY = "entropy"
    REWARD_TYPE_ENTROPY_IMPROVEMENT = "entropy-improvement"
    REWARD_TYPE_ACCURACY = "accuracy"
    REWARD_TYPE_ACCURACY_IMPROVEMENT = "accuracy-improvement"

    # Actor
    LOSS_TYPE_MAX_ADVANTAGE = "max-advantage"
    LOSS_TYPE_SLOW_CHANGE = "online-ppo"

    # Critic
    LOSS_TYPE_MSE = "mse"
    LOSS_TYPE_HUBER = "huber"

    MODEL_ACTOR = "Actor"
    MODEL_CRITIC = "Critic"

    def __init__(self, config:CifarModelConfig, device):
        super().__init__()
        self.config = config
        self.reward_default = None
        self.loss_actor_scaled = 0
        self.loss_critic_scaled = 0
        self.device = device
        self.create_models()

    def get_trainable_parameters(self) -> list:
        parameters = []
        if self.config.encoder_sparsity <= 0:
            parameters_actor = list(self.model_actor.parameters())
            parameters_critic = list(self.model_critic.parameters())
            parameters.extend(parameters_actor)
            parameters.extend(parameters_critic)
        else:
            parameters.extend(self.model_actor.get_trainable_parameters())
            parameters.extend(self.model_critic.get_trainable_parameters())
        return parameters

    def state_dict(self):
        """
        Persist (serialize) trainable parameters - compatible with nn.Module API
        Don't want to serialize model_class, as it's frozen and loaded separately.
        """
        d = {
            "model_actor": self.model_actor.state_dict(),
            "model_critic": self.model_critic.state_dict(),
        }
        return d

    def load_state_dict(self, state_dict):
        """
        Restore (deserialized) trainable parameters - compatible with nn.Module API
        Don't want to serialize model_class, as it's frozen and loaded separately.
        """
        self.model_actor.load_state_dict(
            state_dict["model_actor"]
        )
        self.model_critic.load_state_dict(
            state_dict["model_critic"]
        )

    def create_model_config(self, name:str, input_size:int, output_size:int):
        model_config = SparseActivationDenseModelConfig(
            weight_decay_factor = 0,  # in theory good, but empirically forgets too fast
            name = name,
            nonlinearity = self.config.model_nonlinearity,
            input_layer_norm = True,
            input_dropout = 0,  # bad regularizer
            input_weight_clip = 0.0,
            input_size = input_size,
            hidden_size = self.config.model_hidden_size * self.config.model_hidden_size_factor,
            hidden_size_factor = self.config.model_hidden_size_factor,
            output_size = output_size,
            output_nonlinearity = None,
            layers = self.config.model_layers,
            hidden_dropout = 0,  # bad regularizer 
            bias = True,
        )
        return model_config

    def create_model(self, name, input_size:int, output_size:int) -> SparseDistributedModel:
        logger.info(f"{name} model input:{input_size} output:{output_size}")

        if self.config.encoder_sparsity <= 0:
            # Ablation - dense model
            logger.info(f"Ablation: {name} using dense model.")
            model_config = DenseModelConfig(
                name = name,
                nonlinearity = self.config.model_nonlinearity,
                input_layer_norm = True,
                input_dropout = 0,
                input_weight_clip = 0.0,
                input_size = input_size,
                hidden_size = self.config.model_hidden_size * self.config.model_hidden_size_factor,
                output_size = output_size,
                output_nonlinearity = None,
                layers = self.config.model_layers,
                hidden_dropout = 0,
                bias = True,
            )
            model = DenseModel(model_config)
        else:            
            model_config = self.create_model_config(
                name=name, 
                input_size=input_size, 
                output_size=output_size,
            )
            model = SparseDistributedModel(
                model_configs = [model_config],
                input_key_size = input_size,
                input_value_size = input_size,
                memory_size = self.config.model_hidden_size,
                ensemble_size = self.config.encoder_ensemble_size,
                sparsity = self.config.encoder_sparsity,
            )
        return model

    def create_models(self):

        self.create_classifier()
        encoded_size = self.model_class.get_bias_size()
        bias_size = self.get_bias_size()
        input_size = (encoded_size + bias_size + self.config.num_classes) * self.config.history_size  # input, and bias applied.
        logger.info(f"Encoded obs. size: {encoded_size}")
        logger.info(f"Encoder bias size: {bias_size}")
        logger.info(f"Model input size: {input_size}")

        if self.config.policy_std is None:
            num_terms = 2  # mean + log_std
            logger.info("Using model policy std.")
        else:
            num_terms = 1  # mean 
            logger.info(f"Using constant policy std: {self.config.policy_std}")

        output_size_actor = bias_size * num_terms  # mean, std per bias element
        logger.info(f"Actor action space terms: {num_terms} size: {output_size_actor}")

        self.policy_config = PolicyConfig(
            action_type = PolicyUtil.ACTION_TYPE_CONTINUOUS,    
            policy_std = self.config.policy_std,
        )

        self.model_actor = self.create_model(
            name = "Actor",
            input_size = input_size,
            output_size = output_size_actor,
        )

        self.model_critic = self.create_model(
            name = "Critic",
            input_size = input_size,
            output_size = 1,
        )

    def get_bias_size(self) -> int:
        bias_size = self.model_class.get_bias_size()
        return bias_size
        
    def create_classifier(self):
        config = ResNetConfig(
            num_classes=self.config.num_classes,
        )
        self.model_class = CifarClassifier(
            config, 
            bias_stage=self.config.classifier_bias_stage,
        ).to(self.device)
        self.model_class.eval()

        logger.info(f"Loading conv. model from file: {self.config.classifier_model_file}")
        state_dict = torch.load(self.config.classifier_model_file, weights_only=True)
        self.model_class.load_state_dict(state_dict)

    def do_classifier(
        self,
        image:torch.Tensor, 
        bias:torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():  # Never any grads during RL phase
            logits, encoding = self.model_class(
                x = image,
                bias = bias,
            )
            return logits, encoding

    def do_actor(
        self,
        input:torch.Tensor, 
    ) -> PolicyModelOutput:
        policy_logits, mask = self.do_model(
            model=self.model_actor,
            input=input,
        )

        policy_distribution, policy_sample = PolicyUtil.sample_policy(
            logits = policy_logits,
            config = self.policy_config, 
        )  # produce policy to reach g from x1
        policy_sample_one_hot = PolicyUtil.one_hot_validate(policy_sample)
        output = PolicyModelOutput(
            logits = policy_logits,
            mask = mask,
            distribution = policy_distribution,
            sample = policy_sample,
            sample_one_hot = policy_sample_one_hot,
        )
        return output

    def do_critic(
        self,
        input:torch.Tensor, 
    ) -> torch.Tensor:
        output, mask = self.do_model(
            model = self.model_critic,
            input = input,
        )  # output = [B,1]
        return output.squeeze(1)  # [B]
    
    def do_model(
        self,
        model,
        input:torch.Tensor, 
        key_input:torch.Tensor|None = None,
        active_mask:torch.Tensor|None = None,
        model_index:int = 0,
    ) -> torch.Tensor:
        if key_input is None:
            key_input = input
        if self.config.encoder_sparsity <= 0:
            output = model(input)
            active_mask_output = None
        else:
            output, active_mask_output = model.do_model(
                key_input = key_input, 
                model_input = input, 
                model_index = model_index,
                active_mask = active_mask,
            )
        return output, active_mask_output

    def get_advantage(
        self,
        input_1:torch.Tensor, 
        rewards:torch.Tensor, 
        terminated:torch.Tensor, 
        input_2: torch.Tensor,
    ) -> torch.Tensor:
        v1 = self.do_critic(input_1)  # vN shape [b]

        # handle terminal state v2 should be 0
        # Mask some batch samples if not learning this step - 
        # making advantage zero essentially disables learning.
        with torch.no_grad():
            v2 = self.do_critic(input_2)  # no grad
            v2 = v2 * (1.0 - terminated)  # ie if terminated = 1, then v2 = 0

        returns = rewards + (self.config.discount_factor * v2)
        advantage = returns - v1
        return advantage

    def get_advantage_normalized(self, advantage:torch.Tensor) -> torch.Tensor:
        if self.config.normalize_advantage:
            return advantage
        
        advantage_normalized = normalize_advantage(
            advantage, 
            epsilon = self.config.normalize_advantage_epsilon,
            clamp = self.config.normalize_advantage_clamp,
        )
        return advantage_normalized

    def get_class_distribution_targets(self, class_labels:torch.Tensor) -> torch.Tensor:
        #print(f"class_labels:{class_labels}")
        return class_labels

    def get_class_reward(
            self, 
            class_distribution_predicted_logits:torch.Tensor,
            class_distribution_predicted:torch.Tensor,
            class_distribution_targets:torch.Tensor,
            reward_previous:torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate reward from classification result, without gradients.
                
        :param self: Description
        :param class_distribution_predicted_logits: Description
        :return: Description
        :rtype: Tensor
        """
        if self.config.reward_type == CifarModel.REWARD_TYPE_ACCURACY_IMPROVEMENT:
            class_distribution_rewards, reward_current = self.get_class_accuracy_improvement_reward(
                class_distribution_logits = class_distribution_predicted_logits,  # No grads 
                class_distribution_targets = class_distribution_targets,
                reward_previous = reward_previous,
            )
            self.reward_previous = reward_current  # no longer need reward_previous
        elif self.config.reward_type == CifarModel.REWARD_TYPE_ACCURACY:
            class_distribution_rewards = self.get_class_accuracy_reward(
                class_distribution_logits = class_distribution_predicted_logits,  # No grads 
                class_distribution_targets = class_distribution_targets,
            )
        elif self.config.reward_type == CifarModel.REWARD_TYPE_ENTROPY:
            class_distribution_rewards = self.get_class_entropy_reward(
                class_distribution_logits = class_distribution_predicted_logits,  # No grads
                class_distribution = class_distribution_predicted,
            )
        elif self.config.reward_type == CifarModel.REWARD_TYPE_ENTROPY_IMPROVEMENT:
            class_distribution_rewards, reward_current = self.get_class_entropy_improvement_reward(
                class_distribution_logits = class_distribution_predicted_logits,  # No grads
                class_distribution = class_distribution_predicted,
                reward_previous = reward_previous,
            )
        else:
            raise ValueError("Reward type not recognized.")        
        return class_distribution_rewards

    def get_class_reward_default(self, batch_size:int) ->  torch.Tensor:
        return torch.zeros(batch_size, device = self.device)
     
    def get_class_accuracy_reward(
        self, 
        class_distribution_logits, 
        class_distribution_targets,
    ) -> torch.Tensor:
        """
        Calculates a unit-bounded reward for distribution correctness.
        
        Args:
            logits: [B, C] Raw model outputs.
            targets: [B, C] One-hot encoded target distributions.
            scale: Controls how strictly errors are penalized.
        Returns:
            [B] Reward tensor bounded between ~0.0 and 1.0 per sample.
        """
        cross_entropy = F.cross_entropy(
            class_distribution_logits, 
            class_distribution_targets, 
            reduction='none',  # don't reduce over batch dimension
            #weight = class_weights,
        )
    
        # Map [0, inf) loss to a [1, 0) unit reward scale
        reward = torch.exp(-self.config.reward_scale * cross_entropy)
        return reward

    def get_class_accuracy_improvement_reward(
        self, 
        class_distribution_logits, 
        class_distribution_targets,
        reward_previous,
    ) -> torch.Tensor:
        """
        Calculates a reward based on the improvement over the last step.
        
        Args:
            current_logits: [B, C] Current step raw model outputs.
            targets: [B, C] One-hot encoded target distributions.
            prev_reward: [B] The accuracy reward tensor calculated in the PREVIOUS step.
            scale: Must match the scale used in reward_accuracy.
        Returns:
            [B] Reward tensor bounded between -1.0 and 1.0 per sample.
        """

        reward_current = self.get_class_accuracy_reward(
            class_distribution_logits = class_distribution_logits, 
            class_distribution_targets = class_distribution_targets,
        )

        # Calculate the delta (Current - Previous)
        # If current > previous, reward is positive.
        # Max possible improvement is 1.0 (went from 0.0 reward to 1.0 reward).
        # Max possible regression is -1.0 (went from 1.0 reward to 0.0 reward).
        reward_improvement = reward_current - reward_previous
        return reward_improvement, reward_current

    def get_class_entropy_reward(self, class_distribution_logits, class_distribution):
        # 1. Get both probabilities and log-probabilities safely
        log_probs = F.log_softmax(class_distribution_logits, dim=1)
        
        # 2. Calculate Shannon entropy per batch item: H = -sum(p * log(p))
        entropy = -(class_distribution * log_probs).sum(dim=1)
        
        # 3. Maximum possible entropy for C classes
        max_entropy = torch.log(
            torch.tensor(
                self.config.num_classes, 
                dtype=class_distribution_logits.dtype, 
                device=class_distribution_logits.device,
            )
        )
        
        # 4. Normalize entropy to [0, 1] and invert for the final reward
        # i.e. max entropy / max entropy = 1, 1-1 = 0 reward
        reward = 1.0 - (entropy / max_entropy)
        return reward

    def get_class_entropy_improvement_reward(
        self, 
        class_distribution_logits, 
        class_distribution,
        reward_previous,
    ) -> torch.Tensor:
        """
        Calculates a reward based on the improvement over the last step.
        
        Args:
            current_logits: [B, C] Current step raw model outputs.
            targets: [B, C] One-hot encoded target distributions.
            prev_reward: [B] The accuracy reward tensor calculated in the PREVIOUS step.
            scale: Must match the scale used in reward_accuracy.
        Returns:
            [B] Reward tensor bounded between -1.0 and 1.0 per sample.
        """

        reward_current = self.get_class_entropy_reward(
            class_distribution_logits = class_distribution_logits, 
            class_distribution = class_distribution,
        )

        # 2. Calculate the delta (Current - Previous)
        # If current > previous, reward is positive.
        # Max possible improvement is 1.0 (went from 0.0 reward to 1.0 reward).
        # Max possible regression is -1.0 (went from 1.0 reward to 0.0 reward).
        reward_improvement = reward_current - reward_previous
        return reward_improvement, reward_current

    def update_critic_loss(
        self,
        advantage:torch.Tensor,  # with grads
    ):
        if self.config.loss_critic_type == CifarModel.LOSS_TYPE_MSE:
            loss_critic = advantage.pow(2).mean()
        elif self.config.loss_critic_type == CifarModel.LOSS_TYPE_HUBER:
            loss_critic = F.huber_loss(
                advantage, 
                torch.zeros_like(advantage), 
                delta=self.config.loss_huber_delta,
            )
        else:
            raise ValueError("Critic loss type not recognized.")
        self.loss_critic_scaled = loss_critic * self.config.loss_critic_scale

    def update_actor_loss(
        self,
        previous_policy_data:PolicyData,
        policy_output:PolicyModelOutput,
        advantage:torch.Tensor,
    ):
        if self.config.loss_actor_type == CifarModel.LOSS_TYPE_MAX_ADVANTAGE:        
            policy_metrics = PolicyUtil.get_policy_metrics_continuous(
                policy_distribution = policy_output.distribution,
                policy_sample = policy_output.sample,  # The actual action[s] (after interpretation) [B, A] (0s and 1s)
            )
            loss_actor = PolicyUtil.get_policy_loss_maximize_advantage(
                advantage = advantage,
                joint_log_p = policy_metrics.joint_log_p,  # current model, current input current actions
            )
        elif self.config.loss_actor_type == CifarModel.LOSS_TYPE_SLOW_CHANGE:
            if previous_policy_data is None:
                previous_input = None
                logits = None
            else:
                previous_input = previous_policy_data.model_input
                logits, mask = self.do_model(
                    model=self.model_actor,
                    input=previous_input,
                )
            loss_actor = PolicyUtil.get_policy_loss_limit_ratio(
                policy_config = self.policy_config,
                previous_policy_data = previous_policy_data,
                new_model_logits_on_previous_input = logits,  # with grads; result of new model on previous input
            )
        else:
            raise ValueError("Actor loss type not recognized.")
        self.loss_actor_scaled = loss_actor * self.config.loss_actor_scale

    def get_total_loss(self):
        total_loss = self.loss_critic_scaled + self.loss_actor_scaled
        return total_loss
    