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

Group-normalised advantage (Dave, 2 Nov 2026): `episode(..., learn=True, defer=True)` returns the episode's update
inputs instead of updating; `group_update` then takes one step on a group of episodes (consecutive repeats of one
image, never mixed with other images' episodes) with advantage (G - group mean) / max(group std, floor) per step,
in place of the critic baseline. The critic still regresses on the returns.
"""
import math

import torch

from environment.mnist.mnist_model import MnistModel
from util.observation_history import ObservationHistory, ObservationHistoryConfig


def grid_basis(grid, side=7):
    """Fixed basis mapping a grid x grid action to the side x side token bias: each token takes the value of the
    coarse cell it falls in (nearest-cell upsampling). (grid^2, side^2)."""
    cell = (torch.arange(side) * grid) // side
    idx = (cell[:, None] * grid + cell[None, :]).flatten()
    return torch.nn.functional.one_hot(idx, grid * grid).T.float()


def coarse_basis(net, grid):
    """The fixed action basis for a coarse "where" bias (`grid` x `grid` cells upsampled to the token grid) under
    the net's bias mode: where -> grid_basis; what+where -> the identity on the what part, grid_basis on the where
    part (block diagonal). None when grid is 0 (the actor emits the full bias)."""
    if not grid:
        return None
    side = math.isqrt(net.num_tokens)
    if side * side != net.num_tokens or grid > side:
        raise ValueError(f"coarse where grid {grid} needs a square token grid of side >= {grid}")
    where = grid_basis(grid, side)
    if net.bias_mode == "where":
        return where
    if net.bias_mode == "what+where":
        return torch.block_diag(torch.eye(net.d_model), where)
    raise ValueError("a coarse where grid needs the where or what+where bias")


class RLSTM:
    """
    bias_basis: optional fixed (m, bias_size) matrix; the actor then emits m values and the LTM receives
    action @ bias_basis (e.g. a coarse spatial grid for the "where" bias). The STM observes the m-dim action.
    incremental: each action is a change to the bias, which accumulates over the episode (from 0); the LTM receives
    the cumulative bias, and the observation's bias part is the cumulative bias, so the actor can see and cancel
    earlier moves. Otherwise each action is the bias itself.
    grad_stats: record the gradient norm of each parameter group in last_update (diagnostics, ~8% of the time;
    Dave 25 Nov 2026: off in the stream harness, which never reads them).

    With the "first" mask key the key is the episode's first observation, fixed for the whole episode (the
    first-observation conditioner only updates at step 0, recruitment runs after the episode), so the actor's and
    critic's cliques are encoded once at step 0 and reused at the later steps and in the update (exact).

    momentum_active (Dave 25 Nov 2026): SGD momentum only on the episode's active cells. Plain SGD momentum keeps
    moving the cells of earlier samples after their last loss (the buffer decays by `momentum` per update); here a
    cell's weights and momentum buffer are frozen while it is outside the episode's cliques and resume when it is
    selected again. A cell's parameters: its row of each hidden layer (weight and bias), its column of the next
    layer (so the middle layer moves only between two active cells), its column of the output layer. Shared
    parameters (input layer norm, output bias) update every step as before. With every cell active it is plain SGD.
    """

    def __init__(self, ltm_net, model: MnistModel, steps=4, lr=0.01, momentum=0.5, bias_basis=None,
                 incremental=False, grad_stats=True, momentum_active=False):
        self.ltm = ltm_net          # BiasableTransformer (shared with the LTM model; read-only here)
        self.model = model
        self.bias_basis = bias_basis
        self.incremental = incremental
        self.grad_stats = grad_stats
        self.momentum_active = momentum_active
        self.momentum = momentum
        self.lr = lr
        self.buffers = {}    # momentum buffers of the momentum_active step
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

    def episode(self, x, y=None, learn=False, sample=None, defer=False):
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
        fixed_key = self.model.is_first_observation_input()
        with torch.set_grad_enabled(learn):
            for t in range(T):
                inp = self._input(history)
                ctx = torch.full((B, 1), (T - t) / T) if cfg.step_input else None
                cached = fixed_key and t > 0
                v, kc, ac = self.model.do_critic(inp, ctx, active=active_c[0] if cached else None)
                value_window = torch.cat([value_window[:, 1:], v.detach()[:, None]], dim=1)
                policy, ka, aa = self.model.do_actor(inp, value_window, ctx, active=active_a[0] if cached else None)
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
        if learn and defer:
            result["deferred"] = (inputs, windows, contexts, samples, log_p_old, returns.detach())
            if active_a[0] is not None:
                self.recruited += self.model.recruit(keys_a, active_a, keys_c, active_c)
            self.episodes += 1
        elif learn:
            active = active_a[0].repeat(T, 1) if fixed_key and active_a[0] is not None else None  # step-major rows
            cells = (torch.cat(active_a).unique(), torch.cat(active_c).unique()) if self.momentum_active else None
            self._update(inputs, windows, contexts, samples, log_p_old, values, returns, active, cells)
            if active_a[0] is not None:
                self.recruited += self.model.recruit(keys_a, active_a, keys_c, active_c)
            self.episodes += 1
        return result

    def _update(self, inputs, windows, contexts, samples, log_p_old, values, returns, active=None, cells=None):
        """cells: (actor cells, critic cells) active in the episode, for momentum_active."""
        cfg = self.model.config
        v = torch.stack(values)                       # [T, B], with grads
        advantage = (returns - v.detach())            # [T, B]
        log_p_old = torch.stack(log_p_old)
        inp, win, smp = torch.cat(inputs), torch.cat(windows), torch.cat(samples)
        ctx = torch.cat(contexts) if contexts[0] is not None else None
        stats = {"advantage": float(advantage.mean()), "advantage_abs": float(advantage.abs().mean()),
                 "return_0": float(returns[0].mean()), "value_0": float(v[0].detach().mean())}
        for k in range(cfg.actor_updates):
            log_p = self.model.actor_log_p(inp, win, smp, ctx, active=active)
            loss = self.model.actor_loss(log_p, log_p_old.flatten(), advantage.flatten())
            if k == 0:
                critic_loss = self.model.critic_loss(v.flatten(), returns.flatten())
                stats.update({"actor_loss": float(loss), "critic_loss": float(critic_loss)})
                loss = loss + critic_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if k == 0 and self.grad_stats:
                for name, params in self.param_groups.items():
                    stats[f"grad_{name}"] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                                                 if p.grad is not None)))
            if cells is None:
                self.opt.step()
            else:
                self._step_active(*cells)
        self.last_update = stats

    def group_update(self, deferred, std_floor=1e-3):
        """One update on a group of deferred episodes (see the module docstring). The critic values are recomputed
        from the stored inputs (the parameters may have changed since the episodes ran)."""
        if self.momentum_active:
            raise NotImplementedError("momentum_active with group updates")
        cfg = self.model.config
        T = len(deferred[0][0])

        def cat(k):  # step-major over the group: [T * G, ...]
            return torch.cat([torch.cat([d[k][t] for d in deferred]) for t in range(T)])
        inp, win, smp = cat(0), cat(1), cat(3)
        ctx = cat(2) if deferred[0][2][0] is not None else None
        log_p_old = cat(4)
        returns = torch.cat([d[5] for d in deferred], dim=1)       # [T, G]
        G = returns.shape[1]
        std = returns.std(dim=1, keepdim=True).clamp(min=std_floor) if G > 1 else torch.ones(T, 1)
        advantage = (returns - returns.mean(dim=1, keepdim=True)) / std
        stats = {"advantage": float(advantage.mean()), "advantage_abs": float(advantage.abs().mean()),
                 "return_0": float(returns[0].mean()), "group_return_std": float(returns[0].std()) if G > 1 else 0.0}
        for k in range(cfg.actor_updates):
            log_p = self.model.actor_log_p(inp, win, smp, ctx)
            loss = self.model.actor_loss(log_p, log_p_old, advantage.flatten())
            if k == 0:
                v = self.model.do_critic(inp, ctx)[0].reshape(-1)
                critic_loss = self.model.critic_loss(v, returns.flatten())
                stats.update({"actor_loss": float(loss), "critic_loss": float(critic_loss),
                              "value_0": float(v[:G].detach().mean())})
                loss = loss + critic_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if k == 0 and self.grad_stats:
                for name, params in self.param_groups.items():
                    stats[f"grad_{name}"] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in params
                                                                 if p.grad is not None)))
            self.opt.step()
        self.last_update = stats

    def _cell_masks(self, sparse_model, cells):
        """{parameter: bool mask (broadcastable)} of a sparse model's per-cell parameters for `cells` (see class doc)."""
        dense = sparse_model.get_model(0)
        F, L = dense.config.hidden_size_factor, dense.config.layers
        unit = torch.zeros(dense.config.hidden_size, dtype=torch.bool)
        unit[(cells[:, None] * F + torch.arange(F)).flatten()] = True    # hidden units of the cells
        masks = {}
        for l, fc in enumerate(dense.layers):
            hidden = l < L - 1
            rows = unit[:, None] if hidden else torch.ones(fc.out_features, 1, dtype=torch.bool)
            cols = unit[None, :] if l > 0 else torch.ones(1, fc.in_features, dtype=torch.bool)
            masks[fc.weight] = rows & cols
            if fc.bias is not None and hidden:
                masks[fc.bias] = unit
        return masks

    @torch.no_grad()
    def _step_active(self, actor_cells, critic_cells):
        """SGD with momentum on the active cells only (momentum_active). Same arithmetic as torch.optim.SGD (no
        dampening, nesterov or weight decay) on the parameters it updates; the others keep weights and buffers."""
        masks = {**self._cell_masks(self.model.model_actor, actor_cells),
                 **self._cell_masks(self.model.model_critic, critic_cells)}
        for group in self.opt.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.buffers.get(p)
                mask = masks.get(p)
                if mask is None:   # shared parameter: plain SGD step
                    if buf is None:
                        buf = self.buffers[p] = p.grad.detach().clone()
                    else:
                        buf.mul_(self.momentum).add_(p.grad)
                    p.add_(buf, alpha=-self.lr)
                    continue
                if buf is None:
                    buf = self.buffers[p] = torch.zeros_like(p)
                buf.copy_(torch.where(mask, buf * self.momentum + p.grad, buf))
                p.add_(torch.where(mask, buf, torch.zeros_like(buf)), alpha=-self.lr)

    def param_vector(self, group):
        """A detached flat copy of a parameter group (diagnostics: weight change between two points)."""
        return torch.cat([p.detach().flatten() for p in self.param_groups[group]]).clone()

    def num_values(self):
        return self.model.num_values()
