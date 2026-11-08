"""
Unit tests of the RL STM (environment/mnist/mnist_model.py, single_stream/stm.py), the encoder's idle-cell
recruitment and the CLS model's scheduling. The learning tests run the whole stack on a frozen random transformer.
"""
import copy
import math
import unittest

import torch

from environment.mnist.mnist_model import MnistModel, MnistModelConfig
from model.sparse.sparse_distributed_encoder import SparseDistributedEncoder
from model.transformer import BiasableTransformer
from single_stream.buffers import PriorityBuffer, RingBuffer, part_specs
from single_stream.models import BufferedReplayLTM, CLSModel
from single_stream.stm import RLSTM


def make_stm(steps=3, history=3, hidden=200, sparsity=16, mode="where", **cfg):
    torch.manual_seed(0)
    net = BiasableTransformer(num_tokens=49, token_size=8, num_classes=10, d_model=16, heads=2, bias_mode=mode)
    config = MnistModelConfig(encoder_sparsity=sparsity, history_size=history, model_hidden_size=hidden,
                              num_classes=10, **cfg)
    model = MnistModel(config, encoded_size=net.d_model, bias_size=net.bias_size)
    return net, RLSTM(net, model, steps=steps)


class TestReturnsAndRewards(unittest.TestCase):

    def setUp(self):
        _, self.stm = make_stm()
        self.m = self.stm.model

    def test_terminal_reward_mc_gamma1_is_final_score_everywhere(self):
        scores = [torch.tensor([0.1, 0.5]), torch.tensor([0.2, 0.4]), torch.tensor([0.3, 0.9]), torch.tensor([0.7, 0.2])]
        r = self.m.episode_rewards(scores)
        torch.testing.assert_close(r, torch.tensor([[0., 0.], [0., 0.], [0.7, 0.2]]))
        g = self.m.returns(r, torch.zeros_like(r))
        torch.testing.assert_close(g, torch.tensor([[0.7, 0.2]] * 3))

    def test_improvement_rewards_telescope(self):
        self.m.config.reward_type = "accuracy-improvement"
        scores = [torch.rand(4) for _ in range(5)]
        g = self.m.returns(self.m.episode_rewards(scores), torch.zeros(4, 4))
        torch.testing.assert_close(g[0], scores[-1] - scores[0])
        torch.testing.assert_close(g[2], scores[-1] - scores[2])

    def test_discount_and_td(self):
        self.m.config.discount_factor = 0.5
        r = torch.tensor([[1.], [2.], [4.]])
        torch.testing.assert_close(self.m.returns(r, torch.zeros_like(r)), torch.tensor([[3.], [4.], [4.]]))
        self.m.config.returns = "td"
        v = torch.tensor([[10.], [20.], [30.]])
        torch.testing.assert_close(self.m.returns(r, v), torch.tensor([[11.], [17.], [4.]]))  # terminal after last

    def test_accuracy_reward_is_p_of_label(self):
        logits = torch.randn(3, 10)
        y = torch.tensor([1, 4, 7])
        p = torch.softmax(logits, 1)[torch.arange(3), y]
        torch.testing.assert_close(self.m.get_class_accuracy_reward(logits, y), p)


class TestEpisode(unittest.TestCase):

    def test_shapes_and_unbiased_first_step(self):
        net, stm = make_stm()
        x, y = torch.rand(5, 49 * 8), torch.randint(10, (5,))
        res = stm.episode(x, y)
        self.assertEqual(tuple(res["logits"].shape), (4, 5, 10))
        self.assertEqual(tuple(res["values"].shape), (3, 5))
        self.assertEqual(tuple(res["bias"].shape), (3, 5, 49))
        torch.testing.assert_close(res["logits"][0], net.eval()(x))

    def test_history_holds_latest_observations_and_first(self):
        net, stm = make_stm(steps=4, history=2)
        seen = []
        orig = stm.model.do_critic
        stm.model.do_critic = lambda inp, ctx=None: (seen.append(inp.clone()), orig(inp, ctx))[1]
        x = torch.rand(1, 49 * 8)
        res = stm.episode(x)
        O = stm.model.observation_size
        out0 = net.run(x)
        o0 = torch.cat([out0["features"], torch.zeros(1, 49), out0["logits"]], 1)
        torch.testing.assert_close(seen[0][:, -O:], o0)                    # "first" key after the history
        torch.testing.assert_close(seen[0][:, O:2 * O], o0)               # newest slot at step 0
        torch.testing.assert_close(seen[0][:, :O], torch.zeros(1, O))     # empty slot
        out2 = net.run(x, res["bias"][1])
        o2 = torch.cat([out2["features"], res["bias"][1], out2["logits"]], 1)
        torch.testing.assert_close(seen[2][:, O:2 * O], o2, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(seen[3][:, -O:], o0)                   # first kept after it leaves the window

    def test_evaluation_has_no_side_effects(self):
        net, stm = make_stm(recruit_rate=0.1, recruit_idle_steps=0)
        before = copy.deepcopy([p.detach().clone() for p in stm.model.get_trainable_parameters()])
        rw = stm.model.model_actor.recruit_weight.clone()
        with torch.no_grad():
            stm.episode(torch.rand(7, 49 * 8))
        for a, b in zip(before, stm.model.get_trainable_parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(rw, stm.model.model_actor.recruit_weight, atol=0, rtol=0)
        self.assertEqual(stm.episodes, 0)

    def test_mean_policy_is_deterministic(self):
        _, stm = make_stm()
        x = torch.rand(3, 49 * 8)
        torch.testing.assert_close(stm.episode(x)["logits"], stm.episode(x)["logits"], atol=0, rtol=0)

    def test_learning_changes_parameters(self):
        _, stm = make_stm()
        before = [p.detach().clone() for p in stm.model.get_trainable_parameters()]
        stm.episode(torch.rand(1, 49 * 8), torch.tensor([3]), learn=True)
        changed = sum(int(not torch.equal(a, b)) for a, b in zip(before, stm.model.get_trainable_parameters()))
        self.assertGreater(changed, 0)
        self.assertEqual(stm.episodes, 1)

    def test_update_diagnostics(self):
        _, stm = make_stm()
        grouped = {id(p) for ps in stm.param_groups.values() for p in ps}
        self.assertEqual(grouped, {id(p) for p in stm.model.get_trainable_parameters()})
        self.assertIsNone(stm.last_update)
        stm.episode(torch.rand(1, 49 * 8), torch.tensor([3]), learn=True)
        for group in stm.param_groups:
            self.assertGreater(stm.last_update[f"grad_{group}"], 0)
        self.assertIn("advantage", stm.last_update)


def objects(n, k, protos, g):
    """Synthetic token images: 49 noise tokens of size 8, k of them carrying class prototypes at random positions.
    Returns (x [n, 392], classes [n, k])."""
    x = 0.3 * torch.randn(n, 49, 8, generator=g)
    cls = torch.randint(10, (n, k), generator=g)
    pos = torch.stack([torch.randperm(49, generator=g)[:k] for _ in range(n)])
    for j in range(k):
        x[torch.arange(n), pos[:, j]] += protos[cls[:, j]]
    return x.reshape(n, -1), cls


def trained_ltm(mode):
    """A transformer trained on single-object images (accuracy 1.0). On two-object images labelled by the first
    object it is often wrong, and a bias can fix it: the optimal per-image bias raises mean p(y) from ~0.38 to ~0.83
    (where) / ~0.99 (what) on the 8 images used below."""
    g = torch.Generator().manual_seed(0)
    protos = 2 * torch.randn(10, 8, generator=g)
    torch.manual_seed(0)
    net = BiasableTransformer(num_tokens=49, token_size=8, num_classes=10, d_model=16, heads=2, bias_mode=mode)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(400):
        x, c = objects(64, 1, protos, g)
        loss = torch.nn.functional.cross_entropy(net(x), c[:, 0])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in net.parameters():
        p.requires_grad_(False)
    x, c = objects(8, 2, protos, torch.Generator().manual_seed(5))
    return net.eval(), x, c[:, 0]


class TestLearning(unittest.TestCase):
    """End to end on a frozen trained LTM: the STM must learn biases that raise p(y) on fixed two-object images.
    Tests the sign of the policy gradient, the critic and the plumbing between them."""

    def run_bandit(self, mode, reward, episodes):
        net, x, y = trained_ltm(mode)
        torch.manual_seed(0)
        config = MnistModelConfig(encoder_sparsity=16, history_size=2, model_hidden_size=200, num_classes=10,
                                  reward_type=reward, loss_actor_scale=0.1, policy_std=0.5)
        stm = RLSTM(net, MnistModel(config, encoded_size=net.d_model, bias_size=net.bias_size), steps=2, lr=0.02)
        for i in range(episodes):
            j = i % len(x)
            stm.episode(x[j:j + 1], y[j:j + 1], learn=True)
        res = stm.episode(x, y)
        return res["scores"][0].mean().item(), res["scores"][-1].mean().item(), res

    def test_where_terminal(self):
        unbiased, after, res = self.run_bandit("where", "terminal-accuracy", 2000)
        self.assertGreater(after, unbiased + 0.05, (unbiased, after))
        # the critic tracks the outcome: the last pre-action value is close to the final score on average
        self.assertLess(abs(res["values"][-1].mean().item() - after), 0.15)

    def test_what_improvement(self):
        unbiased, after, _ = self.run_bandit("what", "accuracy-improvement", 2000)
        self.assertGreater(after, unbiased + 0.05, (unbiased, after))


class TestBiasBasis(unittest.TestCase):

    def test_grid_basis_upsamples(self):
        from single_stream.stm import grid_basis
        b = grid_basis(2)
        self.assertEqual(tuple(b.shape), (4, 49))
        torch.testing.assert_close(b.sum(0), torch.ones(49))           # every token in exactly one cell
        m = (torch.tensor([[1., 2., 3., 4.]]) @ b).reshape(7, 7)
        self.assertEqual(m[0, 0].item(), 1.0)
        self.assertEqual(m[6, 6].item(), 4.0)
        torch.testing.assert_close(grid_basis(7), torch.eye(49))

    def test_episode_with_basis(self):
        from single_stream.stm import grid_basis
        torch.manual_seed(0)
        net = BiasableTransformer(num_tokens=49, token_size=8, num_classes=10, d_model=16, heads=2)
        b = grid_basis(3)
        model = MnistModel(MnistModelConfig(encoder_sparsity=16, history_size=2, model_hidden_size=100,
                                            num_classes=10), encoded_size=16, bias_size=9)
        stm = RLSTM(net, model, steps=2, bias_basis=b)
        x = torch.rand(2, 49 * 8)
        res = stm.episode(x, torch.tensor([1, 2]), learn=True)
        self.assertEqual(tuple(res["bias"].shape), (2, 2, 9))
        torch.testing.assert_close(res["logits"][-1], net.eval().run(x, res["bias"][-1] @ b)["logits"])


class TestEpisodeInputs(unittest.TestCase):

    def test_clique_constant_over_episode(self):
        """The clique is chosen from the unbiased first observation only, so it is the same at every step, for the
        actor and the critic, whatever biases the actor applies."""
        _, stm = make_stm(steps=5, history=2)
        x = torch.rand(3, 49 * 8)
        res = stm.episode(x, torch.tensor([1, 2, 3]), learn=True)
        for key in ("active_actor", "active_critic"):
            first = res[key][0].sort(1).values
            for a in res[key][1:]:
                torch.testing.assert_close(a.sort(1).values, first, atol=0, rtol=0)

    def test_step_input(self):
        net, stm = make_stm(steps=3, history=2, step_input=True)
        res = stm.episode(torch.rand(2, 49 * 8), torch.tensor([0, 1]), learn=True)
        self.assertEqual(tuple(res["logits"].shape), (4, 2, 10))

    def test_class_bonus(self):
        classes = torch.tensor([[3, 1], [3, 2], [5, 2], [3, 1], [7, 4]])  # [T + 1, B], step 0 = unbiased
        novel = MnistModel.novel_classes(classes)
        torch.testing.assert_close(novel, torch.tensor([[0., 1.], [1., 0.], [0., 0.], [1., 1.]]))
        _, stm = make_stm(reward_type="accuracy-improvement", class_bonus=0.05)
        scores = [torch.zeros(2)] * 5
        torch.testing.assert_close(stm.model.episode_rewards(scores, classes), 0.05 * novel)


class TestIncremental(unittest.TestCase):

    def test_bias_is_cumulative_and_observed(self):
        net, stm = make_stm(steps=3, history=3)
        stm.incremental = True
        x = torch.rand(2, 49 * 8)
        res = stm.episode(x, torch.tensor([1, 2]), learn=True)
        torch.testing.assert_close(res["bias"], torch.cumsum(res["actions"], 0))
        torch.testing.assert_close(res["logits"][-1], net.eval().run(x, res["bias"][-1])["logits"])

    def test_absolute_by_default(self):
        _, stm = make_stm(steps=2, history=2)
        res = stm.episode(torch.rand(2, 49 * 8))
        torch.testing.assert_close(res["bias"], res["actions"])


class TestRecruitment(unittest.TestCase):

    def test_idle_cells_are_recruited_and_active_cells_are_not(self):
        torch.manual_seed(0)
        enc = SparseDistributedEncoder(input_key_size=20, memory_size=50, sparsity=5, device=torch.device("cpu"),
                                       recruit_rate=0.05, recruit_idle_steps=3)
        x = torch.rand(1, 20)
        first = enc.encode(x)
        winners = set(first.flatten().tolist())
        for _ in range(3):  # idle period not yet over: nothing recruited
            self.assertEqual(enc.recruit(x, enc.encode(x)), 0)
        n = enc.recruit(x, enc.encode(x))
        self.assertEqual(n, 50 - 5)  # every never-active cell, none of the winners
        rows = enc.recruit_weight.abs().sum(1) > 0
        self.assertFalse(any(rows[w] for w in winners))
        for _ in range(200):  # with the same input, recruits eventually displace the original clique
            enc.recruit(x, enc.encode(x))
        self.assertNotEqual(set(enc.encode(x).flatten().tolist()), winners)
        self.assertEqual(enc.num_values(), 50 * 20 + 50)

    def test_off_by_default(self):
        enc = SparseDistributedEncoder(input_key_size=20, memory_size=50, sparsity=5, device=torch.device("cpu"))
        x = torch.rand(1, 20)
        self.assertEqual(enc.recruit(x, enc.encode(x)), 0)
        self.assertEqual(enc.num_values(), 0)


class TestCLSScheduling(unittest.TestCase):

    def test_repeats_and_replay_ratio(self):
        net, stm = make_stm(steps=2, history=2)
        buffer = PriorityBuffer(50, part_specs(torch.rand(1, 49 * 8), torch.zeros(1, dtype=torch.long)))
        ltm = BufferedReplayLTM(net, lr=0.01, buffer=buffer, batch=4, period=4)
        cls = CLSModel(ltm, stm, repeats=3, replay_ratio=2.5)
        for i in range(4):
            cls.observe(torch.rand(1, 49 * 8), torch.tensor([i]))
        # first exposure: buffer empty before the LTM step, so its replay draws are skipped
        self.assertEqual(cls.replay_episodes, 10 - 2)
        self.assertEqual(stm.episodes, 4 * 3 + cls.replay_episodes)
        self.assertEqual(ltm.updates, 1)
        v = cls.predict_variants(torch.rand(6, 49 * 8))
        self.assertEqual(set(v), {"stm", "ltm"})

    def _cls(self, period=4, **kw):
        net, stm = make_stm(steps=2, history=2)
        buffer = PriorityBuffer(50, part_specs(torch.rand(1, 49 * 8), torch.zeros(1, dtype=torch.long)))
        ltm = BufferedReplayLTM(net, lr=0.01, buffer=buffer, batch=4, period=period)
        initial = copy.deepcopy(stm.model)
        factory = lambda: RLSTM(net, copy.deepcopy(initial), steps=2)
        return CLSModel(ltm, stm, stm_factory=factory, **kw), initial

    def test_reset_after_each_sleep(self):
        cls, initial = self._cls(period=4, repeats=2, reset_on_sleep=True)
        first = cls.stm
        for i in range(10):
            cls.observe(torch.rand(1, 49 * 8), torch.tensor([i % 10]))
        self.assertEqual(cls.ltm.updates, 2)
        self.assertEqual(cls.resets, 2)
        self.assertIsNot(cls.stm, first)
        # the STM after the last reset learned only from exposures 9 and 10: 2 x 2 live episodes
        self.assertEqual(cls.stm.episodes, 4)
        self.assertEqual(cls.stm.episodes + cls.stm_episodes_before, cls.live_episodes)
        # a fresh STM starts from the initial weights
        fresh = cls.stm_factory()
        for a, b in zip(fresh.model.get_trainable_parameters(), initial.get_trainable_parameters()):
            torch.testing.assert_close(a, b)

    def test_own_stm_buffer(self):
        specs = part_specs(torch.rand(1, 49 * 8), torch.zeros(1, dtype=torch.long))
        for own in (RingBuffer(3, specs), PriorityBuffer(3, specs, alpha_evict=3, alpha_replay=1)):
            cls, _ = self._cls(period=4, repeats=2, interleave=1, reset_on_sleep=True, stm_buffer=own)
            ltm_written = []
            for i in range(10):
                cls.observe(torch.rand(1, 49 * 8), torch.tensor([i % 10]))
                ltm_written.append(cls.ltm.buffer.written)
            # every live sample goes to both buffers; the STM's own buffer is kept through the sleeps
            self.assertEqual(own.written, 10)
            self.assertEqual(own.size, 3)
            self.assertEqual(cls.ltm.buffer.written, 10)
            self.assertEqual(cls.resets, 2)
            # one interleaved replay per exposure, drawn from the STM's buffer (non-empty from the second write)
            self.assertEqual(cls.replay_episodes, 9)
            self.assertGreater(cls.memory_values(), cls.ltm.memory_values() + cls.stm.num_values())

    def test_trace_does_not_change_training(self):
        xs = torch.rand(10, 49 * 8)
        runs = []
        for traced in (False, True):
            torch.manual_seed(0)
            cls, _ = self._cls(period=4, repeats=6, block=2, stop_p=1.1, interleave=1, reset_on_sleep=True)
            if traced:
                cls.trace = []
            preds = [int(cls.observe(xs[i:i + 1], torch.tensor([i % 10]))) for i in range(10)]
            runs.append((preds, cls.predict(xs), cls))
        self.assertEqual(runs[0][0], runs[1][0])
        torch.testing.assert_close(runs[0][1], runs[1][1])
        trace = runs[1][2].trace
        self.assertEqual(len(trace), 10)
        self.assertEqual(trace[0]["stop"], "cap")
        self.assertEqual(trace[0]["repeats"], 6)
        self.assertEqual([b[0] for b in trace[0]["blocks"]], [2, 4])   # no probe after the last repeat
        self.assertEqual(len(trace[-1]["blocks"][0]), 3)               # probe after the following replay

    def test_factorial_probe_leaves_training_unchanged(self):
        from single_stream import stm_probe
        xs = torch.rand(8, 49 * 8)
        runs = []
        for probed in (False, True):
            torch.manual_seed(0)
            cls, _ = self._cls(period=4, repeats=4, block=2, stop_p=1.1, interleave=1, reset_on_sleep=True)
            preds = []
            for i in range(8):
                x, y = xs[i:i + 1], torch.tensor([i % 10])
                if probed and i in (2, 5):
                    recs = stm_probe.probe(cls, x, y, temperature=4.0, ceiling_repeats=6)
                    self.assertEqual(len(recs), 9)
                    self.assertEqual(recs[-1]["repeats"], 6)
                    self.assertEqual({(r["stm"], r["replay"], r["temperature"]) for r in recs[:8]},
                                     {(a, b, c) for a in ("as_is", "fresh") for b in (True, False) for c in (1.0, 4.0)})
                    self.assertIsNone(cls.trace)
                preds.append(int(cls.observe(x, y)))
            runs.append((preds, cls.predict(xs)))
        self.assertEqual(runs[0][0], runs[1][0])
        torch.testing.assert_close(runs[0][1], runs[1][1])

    def test_oracle_bias_raises_p_and_leaves_model_unchanged(self):
        from single_stream import stm_probe
        torch.manual_seed(0)
        cls, _ = self._cls(period=100, repeats=2)
        x, y = torch.rand(1, 49 * 8), torch.tensor([3])
        cls.observe(x, y)
        before = cls.predict(x)
        params = [q.detach().clone() for q in cls.stm.ltm.parameters()]
        recs = stm_probe.oracle(cls, x, y, scales=(1, 4), steps=50, record=(0, 50))
        self.assertEqual([r["oracle_scale"] for r in recs], [1, 4])
        self.assertAlmostEqual(recs[1]["bias_scale"], 4 * recs[0]["bias_scale"], places=5)
        for r in recs:
            self.assertGreater(r["p_end"], r["p"][0][1])
        self.assertGreaterEqual(recs[1]["p_end"], recs[0]["p_end"] - 1e-6)
        torch.testing.assert_close(cls.predict(x), before)
        for q, q0 in zip(cls.stm.ltm.parameters(), params):
            self.assertTrue(q.requires_grad)
            torch.testing.assert_close(q, q0)

    def test_reward_probe_sets_reward_and_leaves_training_unchanged(self):
        from single_stream import stm_probe
        xs = torch.rand(6, 49 * 8)
        runs = []
        for probed in (False, True):
            torch.manual_seed(0)
            cls, _ = self._cls(period=4, repeats=4, block=2, stop_p=1.1, interleave=1, reset_on_sleep=True)
            reward = cls.stm.model.config.reward_type
            preds = []
            for i in range(6):
                x, y = xs[i:i + 1], torch.tensor([i % 10])
                if probed and i == 3:
                    recs = stm_probe.rewards(cls, x, y, repeats=6, log_p_floor=1e-8, margin_clip=20.0)
                    self.assertEqual([r["reward"] for r in recs], ["accuracy", "log-accuracy", "margin"])
                    self.assertTrue(all(r["repeats"] == 6 for r in recs))
                    self.assertEqual(cls.stm.model.config.reward_type, reward)
                    self.assertIsNone(cls.trace)
                preds.append(int(cls.observe(x, y)))
            runs.append((preds, cls.predict(xs)))
        self.assertEqual(runs[0][0], runs[1][0])
        torch.testing.assert_close(runs[0][1], runs[1][1])

    def test_reward_clips_are_configurable(self):
        import dataclasses
        logits, y = torch.tensor([[0.0, 0.0, 0.0], [0.0, 30.0, 0.0]]), torch.tensor([0, 0])
        _, stm = make_stm(reward_type="log-accuracy")
        stm.model.config = dataclasses.replace(stm.model.config, log_p_floor=math.log(1e-8))
        self.assertAlmostEqual(float(stm.model.step_score(logits, y)[1]), math.log(1e-8), places=4)
        _, stm = make_stm(reward_type="margin")
        stm.model.config = dataclasses.replace(stm.model.config, margin_clip=20.0)
        torch.testing.assert_close(stm.model.step_score(logits, y), torch.tensor([0.0, -20.0]))

    def test_accuracy_log_reward(self):
        import dataclasses
        logits, y = torch.tensor([[2.0, 0.0, -1.0], [0.0, 30.0, 0.0]]), torch.tensor([0, 0])
        _, stm = make_stm(reward_type="accuracy-log")
        floor = math.log(1e-8)
        stm.model.config = dataclasses.replace(stm.model.config, log_p_floor=floor, log_weight=0.5)
        s = stm.model.step_score(logits, y)
        lp = torch.log_softmax(logits[0], 0)[0]
        torch.testing.assert_close(s[0], lp.exp() + 0.5 * (lp - floor) / abs(floor))
        self.assertAlmostEqual(float(s[1]), 1e-8, places=6)   # clipped: the log term is 0, p(y) the floor
        self.assertIn("accuracy-log", MnistModel.REWARD_TYPES)

    def test_no_reset_by_default(self):
        cls, _ = self._cls(period=4, repeats=2)
        first = cls.stm
        for i in range(8):
            cls.observe(torch.rand(1, 49 * 8), torch.tensor([i]))
        self.assertIs(cls.stm, first)
        self.assertEqual(cls.resets, 0)

    def test_adaptive_repeats_stop_and_cap(self):
        cls, _ = self._cls(repeats=8, block=2, stop_p=0.0)
        cls.observe(torch.rand(1, 49 * 8), torch.tensor([3]))
        self.assertEqual(cls.live_episodes, 2)   # stopped after the first block
        cls, _ = self._cls(repeats=8, block=2, stop_p=1.1)
        cls.observe(torch.rand(1, 49 * 8), torch.tensor([3]))
        self.assertEqual(cls.live_episodes, 8)   # never reached: the cap

    def test_interleaved_replay(self):
        cls, _ = self._cls(period=100, repeats=3, interleave=2)
        cls.observe(torch.rand(1, 49 * 8), torch.tensor([1]))   # buffer empty during the first exposure's repeats
        self.assertEqual(cls.replay_episodes, 0)
        cls.observe(torch.rand(1, 49 * 8), torch.tensor([2]))
        self.assertEqual(cls.replay_episodes, 2 * 2)             # before repeats 2 and 3

    def test_stm_delay_is_the_bare_ltm_until_online(self):
        torch.manual_seed(1)
        xs, ys = torch.rand(8, 49 * 8), torch.randint(0, 10, (8,))
        cls, _ = self._cls(period=2, repeats=2, stm_delay=5, reset_on_sleep=True)
        net2, _ = make_stm(steps=2, history=2)   # the same initial LTM weights (make_stm seeds)
        buf2 = PriorityBuffer(50, part_specs(torch.rand(1, 49 * 8), torch.zeros(1, dtype=torch.long)))
        bare = BufferedReplayLTM(net2, lr=0.01, buffer=buf2, batch=4, period=2)
        for i in range(5):
            self.assertFalse(cls.online)
            torch.testing.assert_close(cls.predict(xs), bare.predict(xs))   # offline: the unbiased LTM
            cls.observe(xs[i:i + 1], ys[i:i + 1])
            bare.observe(xs[i:i + 1], ys[i:i + 1])
        self.assertEqual(cls.live_episodes + cls.replay_episodes, 0)
        self.assertEqual(cls.resets, 0)
        self.assertEqual(cls.ltm.updates, bare.updates)
        torch.testing.assert_close(cls.ltm.predict(xs), bare.predict(xs))  # identical LTM training
        self.assertTrue(cls.online)
        cls.observe(xs[5:6], ys[5:6])
        self.assertEqual(cls.live_episodes, 2)

    def test_live_ltm_trains_on_the_live_sample(self):
        net, _ = make_stm()
        buffer = PriorityBuffer(50, part_specs(torch.rand(1, 49 * 8), torch.zeros(1, dtype=torch.long)))
        ltm = BufferedReplayLTM(net, lr=0.1, buffer=buffer, batch=2, period=1, live=True)
        x, y = torch.rand(1, 49 * 8), torch.tensor([7])
        before = torch.softmax(ltm.predict(x), 1)[0, 7]
        for _ in range(20):
            ltm.observe(x, y)
        self.assertEqual(ltm.updates, 20)
        self.assertGreater(float(torch.softmax(ltm.predict(x), 1)[0, 7]), float(before))



class TestWave8(unittest.TestCase):
    """Wave 8 (2 Nov 2026): log-probability and logit-margin rewards, group-normalised advantage, the stream
    without replacement."""

    def test_log_accuracy_and_margin_rewards(self):
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 50.0, 0.0]])
        y = torch.tensor([0, 0])
        _, stm = make_stm(reward_type="log-accuracy")
        s = stm.model.step_score(logits, y)
        torch.testing.assert_close(s[0], torch.log_softmax(logits[0], 0)[0])
        self.assertAlmostEqual(float(s[1]), MnistModel.LOG_P_FLOOR, places=5)    # clipped
        _, stm = make_stm(reward_type="margin")
        s = stm.model.step_score(logits, y)
        torch.testing.assert_close(s, torch.tensor([2.0, -MnistModel.MARGIN_CLIP]))

    def test_group_update(self):
        net, stm = make_stm(reward_type="accuracy")
        x = torch.randn(1, 49 * 8)
        y = torch.tensor([3])
        before = [p.detach().clone() for p in stm.model.get_trainable_parameters()]
        group = [stm.episode(x, y, learn=True, defer=True)["deferred"] for _ in range(4)]
        for p, b in zip(stm.model.get_trainable_parameters(), before):   # deferring does not update
            self.assertTrue(torch.equal(p.detach(), b))
        stm.episode(torch.randn(1, 49 * 8), torch.tensor([1]), learn=True)  # an interleaved update in between
        stm.group_update(group)
        self.assertIn("group_return_std", stm.last_update)
        changed = any(not torch.equal(p.detach(), b) for p, b in zip(stm.model.get_trainable_parameters(), before))
        self.assertTrue(changed)
        self.assertAlmostEqual(stm.last_update["advantage"], 0.0, places=5)   # zero mean within the group

    def test_stream_without_replacement(self):
        import argparse
        from single_stream.stm_frozen import stream_order
        yt = torch.tensor([9] * 16 + [0] * 1000)
        args = argparse.Namespace(seed=0, stream_seed=None, episodes=400, new_fraction=0.25, new_digit=9,
                                  new_order="shuffle")
        o = stream_order(args, yt)
        nines = o[yt[o] == 9].tolist()
        self.assertEqual(len(set(nines[:16])), 16)         # every slot a new image until all are shown
        self.assertEqual(len(set(nines[16:32])), 16)        # then a reshuffled pass
        args.stream_seed = 5
        self.assertFalse(torch.equal(stream_order(args, yt), o))
        args.seed, args.stream_seed = 3, 0                  # the stream depends on the stream seed only
        self.assertTrue(torch.equal(stream_order(args, yt), o))


if __name__ == "__main__":
    unittest.main()
