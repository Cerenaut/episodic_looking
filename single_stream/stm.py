"""
RL STM episode loop (lean loop, Dave 1 Nov 2026: replaces the gym agent/env of environment/cifar for the image
harness; the model code is environment/mnist/mnist_model.py).

An episode is one image (or a batch of images, evaluated independently in parallel) and T steps:

  step 0:   the LTM classifies the image with no bias; observation o_0 = [features, bias 0, logits] enters the
            history (and is the episode's first observation, the default mask key).
  step t:   the critic values the history, V_t; the actor reads the history and the values of the history steps
            and emits a bias a_t; the LTM classifies the image again under a_t; o_{t+1} enters the history.
  output:   the LTM logits after the last bias (step T).

Training (label given): rewards from the per-step scores (MnistModel.episode_rewards), returns over the episode
(terminal after the last action), one SGD step on the actor and critic losses of the whole episode (minibatch =
one episode), then `actor_updates - 1` further clipped-surrogate actor passes, then recruitment. The LTM is frozen
within the episode and never trained here (it learns from replay, outside). Evaluation: no label, no learning, the
policy mean (or a sample) as the bias. Episodes are independent: nothing is carried from one to the next.
"""
import torch

from environment.mnist.mnist_model import MnistModel
from util.observation_history import ObservationHistory, ObservationHistoryConfig


def grid_basis(grid, side=7):
    """Fixed basis mapping a grid x grid action to the side x side token bias: each token takes the value of the
    coarse cell it falls in (nearest-cell upsampling). (grid^2, side^2)."""
    cell = (torch.arange(side) * grid) // side
    idx = (cell[:, None] * grid + cell[None, :]).flatten()
    return torch.nn.functional.one_hot(idx, grid * grid).T.float()


class RLSTM:
    """
    bias_basis: optional fixed (m, bias_size) matrix; the actor then emits m values and the LTM receives
    action @ bias_basis (e.g. a coarse spatial grid for the "where" bias). The STM observes the m-dim action.
    incremental: each action is a change to the bias, which accumulates over the episode (from 0); the LTM receives
    the cumulative bias, and the observation's bias part is the cumulative bias, so the actor can see and cancel
    earlier moves. Otherwise each action is the bias itself.
    """

    def __init__(self, ltm_net, model: MnistModel, steps=4, lr=0.01, momentum=0.5, bias_basis=None,
                 incremental=False):
        self.ltm = ltm_net          # BiasableTransformer (shared with the LTM model; read-only here)
        self.model = model
        self.bias_basis = bias_basis
        self.incremental = incremental
        self.steps = steps
        self.opt = torch.optim.SGD(model.get_trainable_parameters(), lr=lr, momentum=momentum)
        self.episodes = 0
        self.recruited = 0
        self.last_update = None   # diagnostics of the last update (losses, advantages, gradient norms)
        self.param_groups = self._param_groups()

    def _param_groups(self):
        """Named parameter groups for diagnostics: actor_shared (input layer norm and output bias, applied to every
        input whatever its clique), actor_hidden (hidden layer weights and biases), actor_out (output weights), critic."""
        groups = {"actor_shared": [], "actor_hidden": [], "actor_out": [], "critic": []}
        out_layer = f"dense-layer-{self.model.config.model_layers - 1}"
        for name, p in self.model.model_actor.named_parameters():
            if "layer-norm" in name or name.endswith(f"{out_layer}.bias"):
                groups["actor_shared"].append(p)
            elif out_layer in name:
                groups["actor_out"].append(p)
            else:
                groups["actor_hidden"].append(p)
        groups["critic"] = list(self.model.model_critic.parameters())
        return groups

    def observation(self, out, bias):
        return torch.cat([out["features"], bias, out["logits"]], dim=1).detach()

    def _input(self, history):
        x = history.get_tensor_vector()
        if self.model.is_first_observation_input():
            x = torch.cat([x, history.get_first()], dim=1)
        return x

    @torch.no_grad()
    def _ltm(self, x, bias):
        self.ltm.eval()
        if bias is not None and self.bias_basis is not None:
            bias = bias @ self.bias_basis
        return self.ltm.run(x, bias)

    def episode(self, x, y=None, learn=False, sample=None):
        """
        x: [B, ...] LTM inputs; y: [B] labels (training) or None. learn requires y. sample: draw the bias from the
        policy (default: in training, or when eval_bias is "sample"). Returns a dict of per-step diagnostics:
        logits [T + 1, B, C] (0 = unbiased), values [T, B], bias [T, B, bias_size], and with y: scores [T + 1, B],
        rewards [T, B], returns [T, B].
        """
        if learn and y is None:
            raise ValueError("learning needs the label")
        cfg = self.model.config
        if sample is None:
            sample = learn or cfg.eval_bias == MnistModel.EVAL_BIAS_SAMPLE
        B, H, T = len(x), cfg.history_size, self.steps
        history = ObservationHistory(ObservationHistoryConfig(batch_size=B, history_size=H,
                                                              observation_size=self.model.observation_size))
        bias = torch.zeros(B, self.model.bias_size)
        out = self._ltm(x, None)
        obs = self.observation(out, bias)
        if learn:
            self.model.update_input_statistics(obs, history.is_empty())
        history.update(obs)
        value_window = torch.zeros(B, H)

        logits = [out["logits"]]
        inputs, windows, contexts, samples, log_p_old, values = [], [], [], [], [], []
        keys_a, active_a, keys_c, active_c, biases = [], [], [], [], []
        with torch.set_grad_enabled(learn):
            for t in range(T):
                inp = self._input(history)
                ctx = torch.full((B, 1), (T - t) / T) if cfg.step_input else None
                v, kc, ac = self.model.do_critic(inp, ctx)
                value_window = torch.cat([value_window[:, 1:], v.detach()[:, None]], dim=1)
                policy, ka, aa = self.model.do_actor(inp, value_window, ctx)
                a = policy.sample if sample else policy.mean.detach()
                a = a.detach()
                bias = bias + a if self.incremental else a   # bias applied (cumulative if incremental)
                out = self._ltm(x, bias)
                obs = self.observation(out, bias)
                if learn:
                    self.model.update_input_statistics(obs, history.is_empty())
                history.update(obs)

                logits.append(out["logits"])
                inputs.append(inp)
                windows.append(value_window)
                contexts.append(ctx)
                samples.append(a)
                log_p_old.append(policy.distribution.log_prob(a).sum(-1).detach())
                values.append(v)
                biases.append(bias)
                keys_a.append(ka); active_a.append(aa); keys_c.append(kc); active_c.append(ac)

        result = {"logits": torch.stack(logits), "values": torch.stack(values).detach(), "bias": torch.stack(biases),
                  "actions": torch.stack(samples), "active_actor": active_a, "active_critic": active_c}
        if y is None:
            return result
        scores = [self.model.step_score(lg, y) for lg in logits]
        rewards = self.model.episode_rewards(scores, result["logits"].argmax(2))
        returns = self.model.returns(rewards, result["values"])
        result.update({"scores": torch.stack(scores), "rewards": rewards, "returns": returns})
        if learn:
            self._update(inputs, windows, contexts, samples, log_p_old, values, returns)
            if active_a[0] is not None:
                self.recruited += self.model.recruit(keys_a, active_a, keys_c, active_c)
            self.episodes += 1
        return result

    def _update(self, inputs, windows, contexts, samples, log_p_old, values, returns):
        cfg = self.model.config
        v = torch.stack(values)                       # [T, B], with grads
        advantage = (returns - v.detach())            # [T, B]
        log_p_old = torch.stack(log_p_old)
        inp, win, smp = torch.cat(inputs), torch.cat(windows), torch.cat(samples)
        ctx = torch.cat(contexts) if contexts[0] is not None else None
        stats = {"advantage": float(advantage.mean()), "advantage_abs": float(advantage.abs().mean()),
                 "return_0": float(returns[0].mean()), "value_0": float(v[0].detach().mean())}
        for k in range(cfg.actor_updates):
            log_p = self.model.actor_log_p(inp, win, smp, ctx)
            loss = self.model.actor_loss(log_p, log_p_old.flatten(), advantage.flatten())
            if k == 0:
                critic_loss = self.model.critic_loss(v.flatten(), returns.flatten())
                stats.update({"actor_loss": float(loss), "critic_loss": float(critic_loss)})
                loss = loss + critic_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if k == 0:
                for name, params in self.param_groups.items():
                    stats[f"grad_{name}"] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                                                 if p.grad is not None)))
            self.opt.step()
        self.last_update = stats

    def param_vector(self, group):
        """A detached flat copy of a parameter group (diagnostics: weight change between two points)."""
        return torch.cat([p.detach().flatten() for p in self.param_groups[group]]).clone()

    def num_values(self):
        return self.model.num_values()
