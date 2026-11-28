"""Unit tests of the comparison heads and HeadModel (single_stream/heads.py)."""
import types
import unittest

import torch
import torch.nn.functional as F

from model.transformer import BiasableTransformer
from single_stream.buffers import PriorityBuffer, part_specs
from single_stream.heads import FlyHead, HeadModel, KNNHead, SDMLPHead
from single_stream.models import BufferedReplayLTM
from single_stream.run_stream import build_model, build_parser


def make_ltm(period=4):
    torch.manual_seed(0)
    net = BiasableTransformer(num_tokens=49, token_size=8, num_classes=10, d_model=16, heads=2, bias_mode="what")
    buffer = PriorityBuffer(50, part_specs(torch.rand(1, 49 * 8), torch.zeros(1, dtype=torch.long)))
    return BufferedReplayLTM(net, lr=0.01, buffer=buffer, batch=4, period=period)


class TestHeads(unittest.TestCase):

    def test_knn_recalls_stored_samples(self):
        torch.manual_seed(0)
        f, y = torch.randn(20, 16), torch.arange(20) % 10
        head = KNNHead(16, 10, k=1)
        self.assertTrue(head.empty)
        head.fit(f, y)
        self.assertEqual(head.logits(f).argmax(1).tolist(), y.tolist())

    def test_knn_capacity(self):
        head = KNNHead(4, 10, capacity=5)
        head.fit(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))
        self.assertEqual(len(head.labels), 5)

    def test_fly_one_shot(self):
        torch.manual_seed(0)
        f, y = torch.randn(10, 16), torch.arange(10)
        head = FlyHead(16, 10, n_kc=2000, n_response=8, k_frac=0.05, lr=1.0)
        head.fit(f, y)
        self.assertTrue((head.W <= 1).all())
        self.assertGreaterEqual((head.logits(f).argmax(1) == y).float().mean().item(), 0.8)

    def test_sdmlp_learns_and_keeps_constraints(self):
        torch.manual_seed(0)
        f, y = torch.randn(10, 16), torch.arange(10)
        head = SDMLPHead(16, 10, nneurons=200, k=8, lr=0.5)
        for _ in range(200):
            head.fit(f, y)
        self.assertTrue(all((p >= 0).all() for p in head.params))
        self.assertTrue(torch.allclose(head.fc1.weight.norm(dim=1), torch.ones(200), atol=1e-5))
        self.assertGreaterEqual((head.logits(f).argmax(1) == y).float().mean().item(), 0.8)


class TestHeadModel(unittest.TestCase):

    def test_delay_replay_and_reset(self):
        ltm = make_ltm(period=4)
        model = HeadModel(ltm, KNNHead(16, 10), replay=2.0, delay=2)
        x = torch.rand(1, 49 * 8)
        model.observe(x, torch.tensor([3]))
        model.observe(x, torch.tensor([3]))
        self.assertTrue(model.head.empty)  # not online yet: the LTM alone
        model.observe(x, torch.tensor([3]))
        self.assertEqual(len(model.head.labels), 1 + 2)  # live + 2 replay draws
        self.assertEqual(model.predict(x).argmax(1).item(), 3)
        model.observe(x, torch.tensor([3]))  # 4th exposure: the LTM sleeps, the head resets
        self.assertEqual(ltm.updates, 1)
        self.assertEqual(model.resets, 1)
        self.assertTrue(model.head.empty)
        torch.testing.assert_close(model.predict(x), ltm.predict(x))  # empty head: the LTM answers

    def test_no_reset_keeps_head(self):
        model = HeadModel(make_ltm(period=2), KNNHead(16, 10), replay=0.0, reset_on_sleep=False)
        for i in range(4):
            model.observe(torch.rand(1, 49 * 8), torch.tensor([i]))
        self.assertEqual(model.resets, 0)
        self.assertEqual(len(model.head.labels), 4)

    def test_repeats_fit_live_sample(self):
        model = HeadModel(make_ltm(period=100), KNNHead(16, 10), replay=0.0, repeats=3)
        model.observe(torch.rand(1, 49 * 8), torch.tensor([3]))
        self.assertEqual(model.head.labels.tolist(), [3, 3, 3])

    def test_ltm_trains_only_in_sleep(self):
        ltm = make_ltm(period=8)
        before = [p.clone() for p in ltm.net.parameters()]
        model = HeadModel(ltm, SDMLPHead(16, 10, nneurons=50, k=4))
        for i in range(7):
            model.observe(torch.rand(1, 49 * 8), torch.tensor([i % 10]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, ltm.net.parameters())))

    def test_build_model_makes_head_models(self):
        train = types.SimpleNamespace(x=torch.rand(4, 49 * 8), y=torch.zeros(4, dtype=torch.long), num_classes=10)
        for head, cls in (("knn", KNNHead), ("fly", FlyHead), ("sdmlp", SDMLPHead)):
            args = build_parser().parse_args(["--model", "head", "--head", head, "--net", "transformer", "--d-model",
                                              "16", "--heads", "2", "--bias-mode", "what", "--head-repeats", "2",
                                              "--out", "unused"])
            model = build_model(args, train, token_shape=(49, 8))
            self.assertIsInstance(model, HeadModel)
            self.assertIsInstance(model.head, cls)
            self.assertEqual(model.repeats, 2)
    def test_sdmlp_centre(self):
        h = SDMLPHead(16, 2, nneurons=50, k=4, centre=0.5)
        h.observe_input(torch.ones(1, 16))
        self.assertTrue(torch.allclose(h.mu, torch.ones(1, 16)))          # n = 1: the plain mean
        h.observe_input(torch.zeros(1, 16))
        self.assertTrue(torch.allclose(h.mu, torch.full((1, 16), .5)))    # rate max(.5, 1/2)
        h.observe_input(torch.zeros(1, 16))
        self.assertTrue(torch.allclose(h.mu, torch.full((1, 16), .25)))   # then exponential at .5
        self.assertEqual(SDMLPHead(16, 2, nneurons=50, k=4).num_values() + 16, h.num_values())
        # a shared dominant direction: without centring the same neurons win for both classes, with it they differ
        g = torch.Generator().manual_seed(0)
        common = 50 * torch.rand(1, 16, generator=g)                     # cosine across the classes ~.999
        a, b = torch.randn(1, 16, generator=g), torch.randn(1, 16, generator=g)
        xa = common + a + .05 * torch.randn(20, 16, generator=g)
        xb = common + b + .05 * torch.randn(20, 16, generator=g)

        def overlap(head):
            sets = []
            for x in (xa, xb):
                with torch.no_grad():
                    f = x - head.mu if head.centre > 0 else x
                    f = F.relu(f); f = f / f.norm(dim=1, keepdim=True)
                    sets.append(set(torch.topk(head.fc1(f), head.k, dim=1).indices.flatten().tolist()))
            return len(sets[0] & sets[1]) / len(sets[0] | sets[1])
        plain, centred = SDMLPHead(16, 2, nneurons=200, k=8), SDMLPHead(16, 2, nneurons=200, k=8, centre=.01)
        centred.observe_input(torch.cat([xa, xb]))
        self.assertGreater(overlap(plain), 0.5)
        self.assertLess(overlap(centred), overlap(plain) / 2)

    def test_reset_keeps_input_statistics(self):
        model = HeadModel(make_ltm(period=4), SDMLPHead(16, 10, nneurons=50, k=4, centre=.1), replay=0)
        for i in range(5):
            model.observe(torch.rand(1, 49 * 8), torch.tensor([i % 10]))
        self.assertEqual(model.resets, 1)
        self.assertEqual(model.head.n, 5)          # the mean saw every live input, across the reset
        self.assertEqual(model.head.seen, 1)       # the head itself restarted


class TestSideRNG(unittest.TestCase):
    """--stm-rng separate (Dave 27 Nov 2026): the fast learner's random numbers leave the LTM's trajectory as the LTM
    alone's at the same seed (replay draws and continual-backprop re-initialisation included)."""

    BASE = ["--net", "transformer", "--d-model", "16", "--heads", "2", "--bias-mode", "what", "--optimizer", "adamw",
            "--lr", "0.003", "--cbp-rate", "0.2", "--cbp-maturity", "2", "--cbp-no-fold", "--buffer", "priority",
            "--replay-capacity", "40", "--alpha-evict", "3", "--alpha-replay", "1", "--replay-batch", "4",
            "--replay-period", "8", "--replay-steps", "4", "--out", "unused"]
    STM = ["--model", "cls", "--stm-hidden", "64", "--stm-sparsity", "8", "--stm-repeats", "4", "--stm-block", "2",
           "--stm-interleave", "1", "--stm-reward", "accuracy-log", "--sleep-reset-stm", "--stm-delay", "6"]

    def ltm_after(self, extra, steps=40):
        train = types.SimpleNamespace(x=torch.rand(4, 49 * 8), y=torch.zeros(4, dtype=torch.long), num_classes=10)
        torch.manual_seed(0)
        model = build_model(build_parser().parse_args(self.BASE + extra), train, token_shape=(49, 8))
        g = torch.Generator().manual_seed(1)
        for i in range(steps):
            model.observe(torch.rand(1, 49 * 8, generator=g), torch.tensor([(i // 4) % 3]))
        net = model.ltm.net if hasattr(model, "ltm") else model.net
        return [p.detach().clone() for p in net.parameters()]

    def same(self, a, b):
        return all(torch.equal(u, v) for u, v in zip(a, b))

    def test_fast_learners_leave_the_ltm_alone(self):
        alone = self.ltm_after(["--model", "buffered"])
        heads = [["--model", "head", "--head", "knn", "--head-delay", "6"],
                 ["--model", "head", "--head", "fly", "--head-delay", "6", "--head-replay", "2"],
                 ["--model", "head", "--head", "sdmlp", "--head-delay", "6", "--head-no-reset"],
                 self.STM]
        for extra in heads:
            self.assertTrue(self.same(alone, self.ltm_after(extra + ["--stm-rng", "separate"])), extra)
            self.assertTrue(self.same(alone, self.ltm_after(extra + ["--stm-rng", "separate", "--stm-seed", "5"])))
        # shared (the default, as before): the head's replay draws and the STM's draws move the LTM's
        self.assertFalse(self.same(alone, self.ltm_after(heads[0])))
        self.assertFalse(self.same(alone, self.ltm_after(self.STM)))

    def test_stm_seed_varies_the_fast_learner(self):
        train = types.SimpleNamespace(x=torch.rand(4, 49 * 8), y=torch.zeros(4, dtype=torch.long), num_classes=10)
        args = [build_parser().parse_args(self.BASE + ["--model", "head", "--head", "fly", "--fly-response", "4",
                                                       "--stm-seed", str(s)]) for s in (0, 0, 1)]
        heads = [build_model(a, train, token_shape=(49, 8)).head for a in args]
        self.assertTrue(torch.equal(heads[0].R, heads[1].R))
        self.assertFalse(torch.equal(heads[0].R, heads[2].R))


if __name__ == "__main__":
    unittest.main()
