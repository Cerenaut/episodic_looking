"""
Unit tests of the RL STM (environment/mnist/mnist_model.py, single_stream/stm.py), the encoder's idle-cell
recruitment and the CLS model's scheduling. The learning tests run the whole stack on a frozen random transformer.
"""
import copy
import unittest

import torch

from environment.mnist.mnist_model import MnistModel, MnistModelConfig
from model.sparse.sparse_distributed_encoder import SparseDistributedEncoder
from model.transformer import BiasableTransformer
from single_stream.buffers import PriorityBuffer, part_specs
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
        stm.model.do_critic = lambda inp: (seen.append(inp.clone()), orig(inp))[1]
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


if __name__ == "__main__":
    unittest.main()
