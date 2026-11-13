"""Unit tests of the comparison heads and HeadModel (single_stream/heads.py)."""
import types
import unittest

import torch

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
                                              "16", "--heads", "2", "--bias-mode", "what", "--out", "unused"])
            model = build_model(args, train, token_shape=(49, 8))
            self.assertIsInstance(model, HeadModel)
            self.assertIsInstance(model.head, cls)


if __name__ == "__main__":
    unittest.main()
