from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal


@dataclass
class PolicyConfig:
    action_type:str = ""
    max_logit_magnitude:float = 10.0
    policy_std:float|None = None
    policy_log_ratio_limit:float = 20.0
    policy_ratio_limit:float = 0.1

@dataclass
class PolicyMetrics:
    joint_log_p:torch.Tensor|None = None
    entropy:torch.Tensor|None = None

@dataclass
class PolicyData:
    model_input:torch.Tensor|None = None
    policy_sample:torch.Tensor|None = None
    advantage:torch.Tensor|None = None  # or returns
    joint_log_p:torch.Tensor|None = None

class PolicyModelOutput:
    def __init__(
            self,
            logits:torch.Tensor,
            mask:torch.Tensor,
            distribution,
            sample:torch.Tensor,
            sample_one_hot:torch.Tensor,
    ):
        self.logits = logits
        self.mask = mask
        self.distribution = distribution
        self.sample = sample
        self.sample_one_hot = sample_one_hot

class PolicyUtil:

    ACTION_TYPE_BINARY = "binary"
    ACTION_TYPE_CONTINUOUS = "continuous"

    @staticmethod
    def sample_policy(
        logits,
        config:PolicyConfig,
    ):
        #print(f"logits.shape:{logits.shape}")
        logits_clamped = PolicyUtil.clamp_logits(logits, max_logit_magnitude=config.max_logit_magnitude)
        policy_distribution = PolicyUtil.create_distribution(logits_clamped, config)
        policy_sample = policy_distribution.sample()
        return policy_distribution, policy_sample

    @staticmethod
    def clamp_logits(logits:torch.Tensor, max_logit_magnitude:float = 0.0):
        if max_logit_magnitude != 0.0:
            logits = torch.clamp(logits, -max_logit_magnitude, max_logit_magnitude)
        return logits

    @staticmethod
    def create_distribution(
        logits,
        config:PolicyConfig,
    ):
        if config.action_type == PolicyUtil.ACTION_TYPE_BINARY:
            policy_distribution = PolicyUtil.create_distribution_binary(logits)
        elif config.action_type == PolicyUtil.ACTION_TYPE_CONTINUOUS:
            policy_distribution = PolicyUtil.create_distribution_continuous(
                logits,
                std = config.policy_std,
            )
        return policy_distribution

    @staticmethod
    def create_distribution_binary(logits:torch.Tensor):
        # Create distribution object
        # Bernoulli is basically sigmoid, so -5~=0 and 5~=1.
        policy_distribution = Bernoulli(logits=logits)  # Detach after logits
        return policy_distribution

    @staticmethod
    def create_distribution_continuous(
            logits:torch.Tensor, 
            std:float|None = None,
    ):
        """
        If std is not None, it is assumed the logits are all means and the std is constant.

        :param logits: Shape [B,A]
        :type logits: torch.Tensor
        :param std: Description
        :type std: float | None
        """
        if std is None:
            num_actions = logits.shape[1] // 2
            means    = logits[:, :num_actions]
            log_stds = logits[:, num_actions:]  # assume and interpret that model produces log probs
            stds = torch.exp(log_stds)

            #print("logits finite:", torch.isfinite(logits).all())
            #print("means finite:", torch.isfinite(means).all())
            #print("log_stds finite:", torch.isfinite(log_stds).all())
            #print("stds finite:", torch.isfinite(stds).all())            
        else:
            means    = logits  # all logits, no log_stds from model
            stds = torch.ones_like(means) * std

        means = torch.tanh(means)  # tanh: -1 <= x <= 1

        policy_distribution = Normal(means, stds)
        return policy_distribution

    @staticmethod
    def get_policy_metrics(
        logits:torch.Tensor,  # must have grads
        policy_distribution,  # must have grads
        policy_sample:torch.Tensor,
        config:PolicyConfig,
    ) -> PolicyMetrics:

        if config.action_type == PolicyUtil.ACTION_TYPE_BINARY:
            return PolicyUtil.get_policy_metrics_binary(
                logits = logits,  # must have grads
                policy_sample = policy_sample,                
            )
        elif config.action_type == PolicyUtil.ACTION_TYPE_CONTINUOUS:
            return PolicyUtil.get_policy_metrics_binary(
                policy_distribution = policy_distribution,  # must have grads
                policy_sample = policy_sample,                
            )

    @staticmethod
    def get_policy_metrics_binary(
        logits,         # Raw model output [B, A]
        policy_sample:torch.Tensor,  # The actual action[s] (after interpretation) [B, A] (0s and 1s)
    ) -> PolicyMetrics:
        # 1. Single pass for log-probabilities
        log_p1 = F.logsigmoid(logits)    # log(p)
        log_p0 = F.logsigmoid(-logits)   # log(1-p)
        
        # 2. Joint Log-Probability of the EXECUTED action
        # We select based on what actually happened
        bits_log_p = (policy_sample * log_p1) + ((1 - policy_sample) * log_p0)
        joint_log_p = bits_log_p.sum(dim=-1)
        
        # 3. Policy Entropy (analytical, from logits)
        # Use exp() to get the raw probabilities from the logs
        p1 = torch.exp(log_p1)
        p0 = torch.exp(log_p0)
        
        # Entropy per bit: -(p1*log_p1 + p0*log_p0)
        entropy_bits = -(p1 * log_p1 + p0 * log_p0)
        
        # Total entropy for the joint distribution is the sum of bit entropies
        #entropy = entropy_bits.sum(dim=-1)
        entropy = entropy_bits.mean(dim=-1)  # I think I chose mean for invariance to batch size
        return PolicyMetrics(
            joint_log_p = joint_log_p,
            entropy = entropy,
        )

    @staticmethod
    def get_policy_metrics_continuous(
        policy_distribution,
        policy_sample:torch.Tensor,  # The actual action[s] (after interpretation) [B, A] (0s and 1s)
    ) -> PolicyMetrics:
        # Individual log probabilities for each of the 'A' action dimensions: shape [B, A]
        # Joint log probability (sum over action dimensions): shape [B]
        joint_log_p = policy_distribution.log_prob(policy_sample).sum(dim=-1) # Summing over actions for joint probability of action combos --> shape [B]
        
        # Entropy per action dimension: shape [B, A]
        # NB: This is differential entropy
        entropy_per_action = policy_distribution.entropy()

        # Joint entropy: shape [B]
        # Take the mean over the batch to get a single scalar for optimization
        entropy = entropy_per_action.sum(dim=-1).mean()
        return PolicyMetrics(
            joint_log_p = joint_log_p,
            entropy = entropy,
        )

    @staticmethod
    def get_policy_loss_maximize_advantage(
        advantage,  # or returns
        joint_log_p,  # current model, current input current actions
    ):
        # Weight by advantage and take the negative mean for the batch
        policy_loss = -(joint_log_p * advantage).mean()    
        return policy_loss

    @staticmethod
    def get_policy_loss_limit_ratio(
        policy_config:PolicyConfig,
        previous_policy_data:PolicyData,
        new_model_logits_on_previous_input:torch.Tensor,  # with grads; result of new model on previous input
    ):
        # Like PPO
        if previous_policy_data is None:
            # first step only where insufficient data available to compute
            policy_loss = torch.tensor(0.0)
        else:  
            # Calculate: 
            # log_p_new_model: log joint policy probability of new model on previous input and actions
            # log_p_old_model: log joint policy probability of old model on previous input and actions
            # We want these to be similar ie not diverging too fast.
            old_model_joint_log_p = previous_policy_data.joint_log_p
            new_model_logits_previous_input = PolicyUtil.clamp_logits(new_model_logits_on_previous_input)

            if policy_config.action_type == PolicyUtil.ACTION_TYPE_BINARY:
                policy_metrics = PolicyUtil.get_policy_metrics_binary(
                    logits = new_model_logits_previous_input,         # Raw model output [B, A]
                    policy_sample = previous_policy_data.policy_sample,  # The actual action[s] sampled by old model [B, A] (0s and 1s)
                )
                new_model_joint_log_p = policy_metrics.joint_log_p

            elif policy_config.action_type == PolicyUtil.ACTION_TYPE_CONTINUOUS:
                policy_distribution = PolicyUtil.create_distribution_continuous(
                    logits = new_model_logits_previous_input, 
                    std = policy_config.policy_std,
                )
                new_model_joint_log_p = policy_distribution.log_prob(previous_policy_data.policy_sample).sum(dim=-1)   # [B]

                policy_metrics = PolicyUtil.get_policy_metrics_continuous(
                    policy_distribution = policy_distribution,
                    policy_sample = previous_policy_data.policy_sample,  # The actual action[s] sampled by old model [B, A] (0s and 1s)
                )
                new_model_joint_log_p = policy_metrics.joint_log_p



            # ratio = p_new / p_old
            # NB difference in log space is a division
            log_ratio = torch.clamp(
                new_model_joint_log_p - old_model_joint_log_p, 
                min = -policy_config.policy_log_ratio_limit, 
                max =  policy_config.policy_log_ratio_limit,
            )
            ratio = torch.exp(log_ratio)

            surrogate_1 = ratio * previous_policy_data.advantage
            surrogate_2 = torch.clamp(
                ratio, 
                1 - policy_config.policy_ratio_limit, 
                1 + policy_config.policy_ratio_limit,
            ) * previous_policy_data.advantage
            
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

        return policy_loss

    @staticmethod
    def create_policy_data(
        policy_model_input:torch.Tensor,
        policy_model_output:PolicyModelOutput,
        advantage:torch.Tensor,
    ) -> PolicyData:
        policy_sample = policy_model_output.sample.detach().clone()
        policy_metrics = PolicyUtil.get_policy_metrics_continuous(
            policy_distribution = policy_model_output.distribution,
            policy_sample = policy_sample,  # The actual action[s] (after interpretation) [B, A] (0s and 1s)
        )
        policy_data = PolicyData(
            model_input = policy_model_input.detach().clone(),  # no grads
            policy_sample = policy_sample,
            advantage = advantage.detach().clone(),  # no grads
            joint_log_p = policy_metrics.joint_log_p.detach().clone(),
        )
        return policy_data

    @staticmethod
    def one_hot_to_indices(action_mask: torch.Tensor):
        """
        Converts one-hot tensor to integer indices for env.step()
        
        Args:
            actions_mask: [B, A] one-hot tensor
        Returns:
            indices: [B] integer tensor
        """
        # Since it's one-hot, argmax gives us the index of the '1'
        return torch.argmax(action_mask, dim=-1)

    @staticmethod
    def one_hot_validate(actions) -> torch.Tensor:
        """
        Converts multi-binary Bernoulli samples into a valid 1-of-N selection.
        
        Args:
            actions: [B, A] tensor of 0s and 1s from torch.bernoulli
        Returns:
            one_hot: [B, A] tensor where exactly one bit is 1 per batch element.
        """
        B, A = actions.shape
        
        # 1. Handle the "All Zeros" case
        # Create a mask for rows where no bits were sampled
        none_selected = (actions.sum(dim=-1) == 0)
        
        # For rows with all zeros, we "fake" a proposal so the env doesn't crash.
        # We pick a random index for these rows.
        if none_selected.any():
            random_indices = torch.randint(0, A, (none_selected.sum(),), device=actions.device)
            # Inject these into the actions tensor
            actions[none_selected, random_indices] = 1.0

        # 2. Handle the "Multiple Bits" case
        # We need to pick exactly one. Using argmax on the first occurrence 
        # is a standard way to resolve multiple proposals.
        noise = torch.rand(B, A, device=actions.device) * 0.01
        scores = actions + noise
        selected_indices = torch.argmax(scores, dim=-1)
        
        # 3. Convert back to one-hot
        one_hot = torch.nn.functional.one_hot(selected_indices, num_classes=A).float()
        return one_hot
