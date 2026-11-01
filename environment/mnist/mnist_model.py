"""
RL STM model for the single-stream image tasks: a sparse actor (bias policy) and a sparse critic. Adapted from
environment/cifar/cifar_model.py (1 Nov 2026): model construction, sparse init, input conditioning and mask keys
and the reward functions are copied unchanged in substance. Removed: the CIFAR LTM (the LTM is passed in by the
episode loop, single_stream/stm.py, which drives this model), the stage-encoding options and the differentiable
actor. Changed or added:

- Observation of one step: [LTM features (CLS output before the head), bias applied, LTM logits], all detached.
- Critic as the actor's label-free feedback (plan.md section 5, 28 Oct; agreed 1 Nov): the actor's input is the
  critic's input plus the critic's value of each step in the history window (detached). The label is only ever the
  critic's training target, so training and test see the same inputs.
- Rewards: "terminal-accuracy" (exp(-scale x CE) of the final step's LTM output, 0 before; with discount 1 the critic
  predicts the probability-like correctness the episode will end with) and the per-step "accuracy-improvement"
  (copied), plus "accuracy", "entropy" and "entropy-improvement" (copied).
- Actor loss: clipped surrogate over the episode's steps, `actor_updates` passes; one pass equals the plain
  advantage-weighted log-likelihood (ratio 1), the "max-advantage" loss. No advantage normalisation: the minibatch
  is one episode. (cifar_model.get_advantage_normalized returns the raw advantage when normalize_advantage is
  True: the condition looks inverted; not copied.)
- Idle-cell recruitment in the actor's and critic's sparse encoders (SparseDistributedEncoder.recruit).
"""
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.dense import DenseModel, DenseModelConfig
from model.sparse.sparse_dense_model import SparseActivationDenseModelConfig
from model.sparse.sparse_distributed_model import SparseDistributedModel
from util.input_conditioner import InputConditioner
from util.reinforcement_learning.policy_util import PolicyConfig, PolicyModelOutput, PolicyUtil

logger = logging.getLogger(__name__)


@dataclass
class MnistModelConfig:

    encoder_ensemble_size: int = 1
    encoder_sparsity: int = 32

    history_size: int = 4

    model_hidden_size: int = 1000
    model_hidden_size_factor: int = 1
    model_nonlinearity: str = "leaky-relu"
    model_layers: int = 3

    discount_factor: float = 1.0
    returns: str = "mc"  # "mc" (Monte Carlo over the episode) or "td" (one-step bootstrapped)

    reward_scale: float = 1.0
    reward_type: str = "terminal-accuracy"

    loss_actor_scale: float = 0.01
    loss_critic_scale: float = 1.0
    loss_huber_delta: float = 1.0
    loss_critic_type: str = "huber"
    actor_updates: int = 1
    policy_ratio_limit: float = 0.1

    num_classes: int = 0
    policy_std: float | None = 0.5
    eval_bias: str = "mean"  # "mean" or "sample" at evaluation
    actor_value_input: bool = True  # the critic's values of the history steps are part of the actor's input

    # Input conditioning and mask key (copied from CifarModelConfig; see there)
    input_conditioning: str = "none"
    input_conditioning_target: str = "mask"
    mask_key: str = "first"
    mask_key_bias_weight: float = 1.0
    mask_key_logit_weight: float = 1.0
    input_stats_burnin_samples: int = 8000
    input_stats_freeze: bool = True
    input_stats_momentum: float = 0.001

    stm_layer_norm_affine: bool = True
    stm_output_bias: bool = True
    stm_critic_sparse_init: bool = False
    stm_actor_sparse_init: bool = False  # same scaling for the actor (its output layer also sees only k inputs)

    # Idle-cell recruitment for the clique selection (SparseDistributedEncoder); 0 = off
    recruit_rate: float = 0.0
    recruit_idle_steps: int = 10000
    recruit_prob: float = 1.0
    seed: int = 0


class MnistModel:

    REWARD_TYPE_TERMINAL_ACCURACY = "terminal-accuracy"
    REWARD_TYPE_ACCURACY = "accuracy"
    REWARD_TYPE_ACCURACY_IMPROVEMENT = "accuracy-improvement"
    REWARD_TYPE_ENTROPY = "entropy"
    REWARD_TYPE_ENTROPY_IMPROVEMENT = "entropy-improvement"
    REWARD_TYPES = [REWARD_TYPE_TERMINAL_ACCURACY, REWARD_TYPE_ACCURACY, REWARD_TYPE_ACCURACY_IMPROVEMENT,
                    REWARD_TYPE_ENTROPY, REWARD_TYPE_ENTROPY_IMPROVEMENT]

    LOSS_TYPE_MSE = "mse"
    LOSS_TYPE_HUBER = "huber"

    RETURNS_MC = "mc"
    RETURNS_TD = "td"

    CONDITIONING_TARGET_MASK = "mask"
    CONDITIONING_TARGET_BOTH = "both"
    MASK_KEY_HISTORY = "history"
    MASK_KEY_CURRENT = "current"
    MASK_KEY_FIRST = "first"

    EVAL_BIAS_SAMPLE = "sample"
    EVAL_BIAS_MEAN = "mean"

    def __init__(self, config: MnistModelConfig, encoded_size: int, bias_size: int, device=None):
        if config.reward_type not in MnistModel.REWARD_TYPES:
            raise ValueError(f"Unknown reward type: {config.reward_type}")
        if config.returns not in (MnistModel.RETURNS_MC, MnistModel.RETURNS_TD):
            raise ValueError(f"Unknown returns: {config.returns}")
        self.config = config
        self.device = device if device is not None else torch.device("cpu")
        self.encoded_size = encoded_size
        self.bias_size = bias_size
        self.create_models()

    # Construction (copied from CifarModel, sizes passed in) ----------------------------------------------------

    def get_trainable_parameters(self) -> list:
        if self.config.encoder_sparsity <= 0:
            return list(self.model_actor.parameters()) + list(self.model_critic.parameters())
        return self.model_actor.get_trainable_parameters() + self.model_critic.get_trainable_parameters()

    def create_model_config(self, name: str, input_size: int, output_size: int):
        return SparseActivationDenseModelConfig(
            weight_decay_factor=0,
            name=name,
            nonlinearity=self.config.model_nonlinearity,
            input_layer_norm=True,
            input_layer_norm_affine=self.config.stm_layer_norm_affine,
            input_dropout=0,
            input_weight_clip=0.0,
            input_size=input_size,
            hidden_size=self.config.model_hidden_size * self.config.model_hidden_size_factor,
            hidden_size_factor=self.config.model_hidden_size_factor,
            output_size=output_size,
            output_nonlinearity=None,
            layers=self.config.model_layers,
            hidden_dropout=0,
            bias=True,
            output_bias=self.config.stm_output_bias,
        )

    def create_model(self, name, input_size: int, output_size: int, key_size: int | None = None, seed: int = 0):
        logger.info(f"{name} model input:{input_size} output:{output_size}")
        if self.config.encoder_sparsity <= 0:
            logger.info(f"Ablation: {name} using dense model.")
            return DenseModel(DenseModelConfig(
                name=name, nonlinearity=self.config.model_nonlinearity, input_layer_norm=True, input_dropout=0,
                input_weight_clip=0.0, input_size=input_size,
                hidden_size=self.config.model_hidden_size * self.config.model_hidden_size_factor,
                output_size=output_size, output_nonlinearity=None, layers=self.config.model_layers,
                hidden_dropout=0, bias=True))
        return SparseDistributedModel(
            model_configs=[self.create_model_config(name=name, input_size=input_size, output_size=output_size)],
            input_key_size=input_size if key_size is None else key_size,
            input_value_size=input_size,
            memory_size=self.config.model_hidden_size,
            ensemble_size=self.config.encoder_ensemble_size,
            sparsity=self.config.encoder_sparsity,
            recruit_rate=self.config.recruit_rate,
            recruit_idle_steps=self.config.recruit_idle_steps,
            recruit_prob=self.config.recruit_prob,
            seed=seed,
        )

    def create_models(self):
        self.observation_part_sizes = [self.encoded_size, self.bias_size, self.config.num_classes]
        self.observation_size = sum(self.observation_part_sizes)
        self.critic_input_size = self.observation_size * self.config.history_size
        if self.is_first_observation_input():
            self.critic_input_size += self.observation_size
        self.value_input_size = self.config.history_size if self.config.actor_value_input else 0
        key_size = self.create_input_conditioning()

        num_terms = 2 if self.config.policy_std is None else 1  # mean (+ log std)
        self.policy_config = PolicyConfig(action_type=PolicyUtil.ACTION_TYPE_CONTINUOUS,
                                          policy_std=self.config.policy_std,
                                          policy_ratio_limit=self.config.policy_ratio_limit)
        history_input = self.observation_size * self.config.history_size
        self.model_actor = self.create_model("Actor", history_input + self.value_input_size,
                                             self.bias_size * num_terms, key_size, seed=self.config.seed)
        self.model_critic = self.create_model("Critic", history_input, 1, key_size, seed=self.config.seed + 1)
        if self.config.stm_critic_sparse_init:
            self.apply_sparse_init(self.model_critic)
        if self.config.stm_actor_sparse_init:
            self.apply_sparse_init(self.model_actor)

    def apply_sparse_init(self, model: SparseDistributedModel):
        mlp = model.get_model(0)
        scale = (self.config.model_hidden_size / self.config.encoder_sparsity) ** 0.5
        with torch.no_grad():
            for layer in mlp.layers[1:]:
                layer.weight.mul_(scale)
            for layer in mlp.layers[:-1]:
                layer.bias.zero_()

    def is_first_observation_input(self) -> bool:
        return self.config.mask_key == MnistModel.MASK_KEY_FIRST

    def create_input_conditioning(self) -> int:
        """Conditioning of the STM input (see CifarModelConfig). Returns the size of the sparse mask's key."""
        if self.config.input_conditioning_target not in (MnistModel.CONDITIONING_TARGET_MASK,
                                                         MnistModel.CONDITIONING_TARGET_BOTH):
            raise ValueError(f"Unknown input conditioning target: {self.config.input_conditioning_target}")
        if self.config.mask_key not in (MnistModel.MASK_KEY_HISTORY, MnistModel.MASK_KEY_CURRENT,
                                        MnistModel.MASK_KEY_FIRST):
            raise ValueError(f"Unknown mask key: {self.config.mask_key}")

        def conditioner():
            return InputConditioner(
                part_sizes=self.observation_part_sizes,
                method=self.config.input_conditioning,
                burnin_samples=self.config.input_stats_burnin_samples,
                freeze=self.config.input_stats_freeze,
                momentum=self.config.input_stats_momentum,
                device=self.device)
        self.input_conditioner_all = conditioner()
        self.input_conditioner_first = conditioner()

        O, E, Bs = self.observation_size, self.encoded_size, self.bias_size
        key_weight = torch.ones(O, device=self.device)
        key_weight[E:E + Bs] = self.config.mask_key_bias_weight
        key_weight[E + Bs:] = self.config.mask_key_logit_weight
        self.mask_key_weight = key_weight
        if self.config.mask_key == MnistModel.MASK_KEY_HISTORY:
            return O * self.config.history_size
        return O

    @torch.no_grad()
    def update_input_statistics(self, observation: torch.Tensor, is_first: torch.Tensor):
        """observation: [B, O] entering the history in training; is_first: [B] bool, first of its episode."""
        for conditioner, x in ((self.input_conditioner_all, observation),
                               (self.input_conditioner_first, observation[is_first])):
            conditioner.update(x)

    def prepare_input(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input: [B, history_size * O] history vector, followed by the first observation [B, O] for the "first" key.
        Returns (mask key, model input). Copied from CifarModel.prepare_input.
        """
        B = input.shape[0]
        T, O = self.config.history_size, self.observation_size
        history = input[:, :T * O].reshape(B, T, O)
        filled = (history != 0).any(dim=-1, keepdim=True)  # empty slots stay zero
        history_conditioned = self.input_conditioner_all(history) * filled
        if self.config.input_conditioning_target == MnistModel.CONDITIONING_TARGET_BOTH:
            model_input = history_conditioned.reshape(B, T * O)
        else:
            model_input = input[:, :T * O]
        if self.config.mask_key == MnistModel.MASK_KEY_HISTORY:
            key = (history_conditioned * self.mask_key_weight).reshape(B, T * O)
        elif self.config.mask_key == MnistModel.MASK_KEY_CURRENT:
            key = history_conditioned[:, -1] * self.mask_key_weight
        else:
            first = input[:, T * O:]
            assert first.shape[1] == O, "the first-observation key needs the first observation after the history"
            key = self.input_conditioner_first(first) * self.mask_key_weight
        return key, model_input

    # Forward passes ---------------------------------------------------------------------------------------------

    def do_model(self, model, input: torch.Tensor, extra: torch.Tensor | None = None):
        """
        input: critic input (history vector, + first observation). extra: [B, n] appended to the model input only
        (not to the mask key), e.g. the actor's value inputs. Returns (output, key, active indices or None).
        """
        key, model_input = self.prepare_input(input)
        if extra is not None:
            model_input = torch.cat([model_input, extra], dim=1)
        if self.config.encoder_sparsity <= 0:
            return model(model_input), key, None
        active_indices = model.encode(key)
        active_mask = model.get_model(0).create_active_mask(model_input, active_indices=active_indices)
        output, _ = model.do_model(key_input=key, model_input=model_input, active_mask=active_mask)
        return output, key, active_indices

    def do_actor(self, input: torch.Tensor, values: torch.Tensor | None):
        logits, key, active = self.do_model(self.model_actor, input,
                                            extra=values if self.config.actor_value_input else None)
        distribution, sample = PolicyUtil.sample_policy(logits=logits, config=self.policy_config)
        mean = PolicyUtil.get_policy_mean_continuous(
            logits=PolicyUtil.clamp_logits(logits, max_logit_magnitude=self.policy_config.max_logit_magnitude),
            std=self.policy_config.policy_std)
        output = PolicyModelOutput(logits=logits, mask=None, distribution=distribution, sample=sample,
                                   sample_one_hot=None, mean=mean)
        return output, key, active

    def actor_log_p(self, input: torch.Tensor, values: torch.Tensor | None, sample: torch.Tensor):
        """Joint log-probability of `sample` under the current actor (with grads)."""
        output, _, _ = self.do_actor(input, values)
        return output.distribution.log_prob(sample).sum(dim=-1)

    def do_critic(self, input: torch.Tensor):
        output, key, active = self.do_model(self.model_critic, input)
        return output.squeeze(1), key, active

    @torch.no_grad()
    def recruit(self, actor_keys, actor_active, critic_keys, critic_active):
        """One recruitment update per step key (training only). Returns the cells recruited."""
        if self.config.encoder_sparsity <= 0:
            return 0
        n = 0
        for model, keys, active in ((self.model_actor, actor_keys, actor_active),
                                    (self.model_critic, critic_keys, critic_active)):
            for k, a in zip(keys, active):
                model.record_active(a)
                n += model.recruit(k, a)
        return n

    def num_values(self) -> int:
        n = sum(p.numel() for p in self.get_trainable_parameters())
        if self.config.encoder_sparsity > 0:
            n += self.model_actor.num_values() + self.model_critic.num_values()
        n += sum(b.numel() for c in (self.input_conditioner_all, self.input_conditioner_first)
                 for b in (c.mean, c.part_sq_norm)) if self.input_conditioner_all.uses_statistics() else 0
        return n

    # Rewards (copied from CifarModel) ---------------------------------------------------------------------------

    def get_class_accuracy_reward(self, logits, targets) -> torch.Tensor:
        """Unit-bounded correctness: exp(-scale x CE), i.e. p(y)^scale. [B]"""
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        return torch.exp(-self.config.reward_scale * cross_entropy)

    def get_class_entropy_reward(self, logits) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(self.config.num_classes)))
        return 1.0 - entropy / max_entropy

    def step_score(self, logits, targets) -> torch.Tensor:
        """The un-differenced per-step quantity the reward is built from."""
        if self.config.reward_type in (MnistModel.REWARD_TYPE_ENTROPY, MnistModel.REWARD_TYPE_ENTROPY_IMPROVEMENT):
            return self.get_class_entropy_reward(logits)
        return self.get_class_accuracy_reward(logits, targets)

    def episode_rewards(self, scores: list[torch.Tensor]) -> torch.Tensor:
        """
        scores: T + 1 per-step scores [B] of the LTM output after each bias (index 0 = unbiased, before any action).
        Returns rewards [T, B] for the T actions.
        """
        s = torch.stack(scores)  # [T + 1, B]
        rt = self.config.reward_type
        if rt in (MnistModel.REWARD_TYPE_ACCURACY_IMPROVEMENT, MnistModel.REWARD_TYPE_ENTROPY_IMPROVEMENT):
            return s[1:] - s[:-1]
        if rt == MnistModel.REWARD_TYPE_TERMINAL_ACCURACY:
            r = torch.zeros_like(s[1:])
            r[-1] = s[-1]
            return r
        return s[1:]

    # Losses -----------------------------------------------------------------------------------------------------

    def returns(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        rewards, values: [T, B] (values = critic estimate before each action, detached). The episode ends after the
        last action (terminal value 0). mc: G_t = r_t + gamma G_{t+1}; td: r_t + gamma V_{t+1}.
        """
        gamma = self.config.discount_factor
        T = rewards.shape[0]
        if self.config.returns == MnistModel.RETURNS_TD:
            v_next = torch.cat([values[1:], torch.zeros_like(values[:1])])
            return rewards + gamma * v_next
        out = torch.zeros_like(rewards)
        g = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            g = rewards[t] + gamma * g
            out[t] = g
        return out

    def critic_loss(self, values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.config.loss_critic_type == MnistModel.LOSS_TYPE_MSE:
            loss = (targets - values).pow(2).mean()
        elif self.config.loss_critic_type == MnistModel.LOSS_TYPE_HUBER:
            loss = F.huber_loss(values, targets, delta=self.config.loss_huber_delta)
        else:
            raise ValueError("Critic loss type not recognized.")
        return loss * self.config.loss_critic_scale

    def actor_loss(self, log_p: torch.Tensor, log_p_old: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """Clipped surrogate (PPO-style); at ratio 1 its gradient is the advantage-weighted log-likelihood's."""
        ratio = torch.exp(torch.clamp(log_p - log_p_old, -20, 20))
        eps = self.config.policy_ratio_limit
        surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage)
        return -surrogate.mean() * self.config.loss_actor_scale
