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


class RLSTM:

    def __init__(self, ltm_net, model: MnistModel, steps=4, lr=0.01, momentum=0.5):
        self.ltm = ltm_net          # BiasableTransformer (shared with the LTM model; read-only here)
        self.model = model
        self.steps = steps
        self.opt = torch.optim.SGD(model.get_trainable_parameters(), lr=lr, momentum=momentum)
        self.episodes = 0
        self.recruited = 0

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
        inputs, windows, samples, log_p_old, values = [], [], [], [], []
        keys_a, active_a, keys_c, active_c, biases = [], [], [], [], []
        with torch.set_grad_enabled(learn):
            for t in range(T):
                inp = self._input(history)
                v, kc, ac = self.model.do_critic(inp)
                value_window = torch.cat([value_window[:, 1:], v.detach()[:, None]], dim=1)
                policy, ka, aa = self.model.do_actor(inp, value_window)
                a = policy.sample if sample else policy.mean.detach()
                a = a.detach()
                out = self._ltm(x, a)
                obs = self.observation(out, a)
                if learn:
                    self.model.update_input_statistics(obs, history.is_empty())
                history.update(obs)

                logits.append(out["logits"])
                inputs.append(inp)
                windows.append(value_window)
                samples.append(a)
                log_p_old.append(policy.distribution.log_prob(a).sum(-1).detach())
                values.append(v)
                biases.append(a)
                keys_a.append(ka); active_a.append(aa); keys_c.append(kc); active_c.append(ac)

        result = {"logits": torch.stack(logits), "values": torch.stack(values).detach(), "bias": torch.stack(biases)}
        if y is None:
            return result
        scores = [self.model.step_score(lg, y) for lg in logits]
        rewards = self.model.episode_rewards(scores)
        returns = self.model.returns(rewards, result["values"])
        result.update({"scores": torch.stack(scores), "rewards": rewards, "returns": returns})
        if learn:
            self._update(inputs, windows, samples, log_p_old, values, returns)
            if active_a[0] is not None:
                self.recruited += self.model.recruit(keys_a, active_a, keys_c, active_c)
            self.episodes += 1
        return result

    def _update(self, inputs, windows, samples, log_p_old, values, returns):
        cfg = self.model.config
        v = torch.stack(values)                       # [T, B], with grads
        advantage = (returns - v.detach())            # [T, B]
        log_p_old = torch.stack(log_p_old)
        inp, win, smp = torch.cat(inputs), torch.cat(windows), torch.cat(samples)
        for k in range(cfg.actor_updates):
            log_p = self.model.actor_log_p(inp, win, smp)
            loss = self.model.actor_loss(log_p, log_p_old.flatten(), advantage.flatten())
            if k == 0:
                loss = loss + self.model.critic_loss(v.flatten(), returns.flatten())
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

    def num_values(self):
        return self.model.num_values()
