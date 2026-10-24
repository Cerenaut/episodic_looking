"""Run: python -m unittest discover -s single_stream/tests -t ."""
import itertools
import unittest
from collections import Counter

import numpy as np
import torch

from single_stream import metrics
from single_stream.models import ContinualBackprop, LTMOnly, MLP, ReplayLTM
from single_stream.stream import ClusteredStream, run_lengths


def groups_of(sizes):
    return np.concatenate([np.full(s, g) for g, s in enumerate(sizes)])


class TestClusteredStream(unittest.TestCase):

    def test_every_sample_exactly_once(self):
        groups = groups_of([37, 50, 13, 64])
        for c in (1, 4, 16, 64, 128):
            ep = ClusteredStream(groups, c, np.random.default_rng(c)).epoch()
            self.assertEqual(sorted(ep.order.tolist()), list(range(len(groups))))
            self.assertEqual(ep.cluster_lengths.sum(), len(groups))

    def test_clusters_are_contiguous_single_group_and_truncated(self):
        groups = groups_of([37, 50, 13])
        c = 16
        ep = ClusteredStream(groups, c, np.random.default_rng(0)).epoch()
        start = 0
        for g, length in zip(ep.cluster_groups, ep.cluster_lengths):
            self.assertTrue(np.all(groups[ep.order[start:start + length]] == g))
            start += length
        # per group: full clusters of c and one truncated tail
        for g, size in enumerate([37, 50, 13]):
            lengths = sorted(ep.cluster_lengths[ep.cluster_groups == g].tolist())
            expected = sorted([c] * (size // c) + ([size % c] if size % c else []))
            self.assertEqual(lengths, expected)

    def test_no_repeat_when_c_above_one(self):
        groups = groups_of([500] * 10)
        for seed in range(20):
            ep = ClusteredStream(groups, 16, np.random.default_rng(seed)).epoch()
            cg = ep.cluster_groups
            for i in np.flatnonzero(cg[1:] == cg[:-1]):
                # a repeat is allowed only when no other group had clusters left
                self.assertTrue(np.all(cg[i:] == cg[i]), f"seed {seed}: repeat at cluster {i} with others left")

    def test_c1_is_uniform_shuffle(self):
        # groups of sizes 2 and 2: all 6 group sequences equally likely under a uniform shuffle
        groups = groups_of([2, 2])
        rng = np.random.default_rng(1)
        stream = ClusteredStream(groups, 1, rng)
        counts = Counter(tuple(groups[stream.epoch().order]) for _ in range(12000))
        self.assertEqual(len(counts), 6)
        for v in counts.values():
            self.assertAlmostEqual(v / 12000, 1 / 6, delta=0.02)
        # and each sample uniform over positions
        positions = np.zeros((4, 4))
        for _ in range(8000):
            order = stream.epoch().order
            positions[order, np.arange(4)] += 1
        np.testing.assert_allclose(positions / 8000, 0.25, atol=0.03)

    def test_c1_allows_repeats(self):
        groups = groups_of([500] * 10)
        ep = ClusteredStream(groups, 1, np.random.default_rng(0)).epoch()
        repeats = np.mean(groups[ep.order][1:] == groups[ep.order][:-1])
        self.assertAlmostEqual(repeats, 0.1, delta=0.02)

    def test_run_lengths(self):
        self.assertEqual(run_lengths([5, 5, 1, 1, 1, 5]).tolist(), [2, 3, 1])
        self.assertEqual(run_lengths([]).tolist(), [])


class TestMetrics(unittest.TestCase):

    def test_schedule(self):
        ckpts, grid = metrics.checkpoint_schedule(1000, 2, 5, 4, step=1)
        self.assertEqual(sorted(grid), [250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        self.assertEqual(ckpts[0], 1)
        self.assertEqual(ckpts[-1], 2000)
        self.assertTrue(set(grid) <= set(ckpts))

    def test_log2_bins(self):
        self.assertEqual(metrics.log2_bin([0, 1, 2, 3, 4, 7, 8]).tolist(), [0, 1, 2, 2, 3, 3, 4])

    def test_balanced_group_score(self):
        # 3 groups = 3 labels, 10 samples each
        perfect = np.diag([10, 10, 10])
        np.testing.assert_allclose(metrics.balanced_group_score(perfect, [0, 1, 2]), 1.0)
        # always predict label 0: recall 1/0/0, specificity 0/1/1 -> 0.5 everywhere
        always0 = np.array([[10, 0, 0], [10, 0, 0], [10, 0, 0]])
        np.testing.assert_allclose(metrics.balanced_group_score(always0, [0, 1, 2]), 0.5)
        # two groups sharing a label (CIFAR coarse targets): specificity only counts the other label's groups
        shared = np.array([[10, 0], [10, 0], [5, 5]])
        np.testing.assert_allclose(metrics.balanced_group_score(shared, [0, 0, 1]), [0.75, 0.75, 0.75])

    def test_group_confusion_and_labels(self):
        preds = np.array([0, 1, 1, 2])
        groups = np.array([5, 5, 7, 7])
        np.testing.assert_array_equal(metrics.group_confusion(preds, groups, [5, 7], 3), [[1, 1, 0], [0, 1, 1]])
        np.testing.assert_array_equal(metrics.group_labels(groups, np.array([0, 0, 1, 1]), [5, 7]), [0, 1])
        with self.assertRaises(ValueError):
            metrics.group_labels(groups, np.array([0, 1, 1, 1]), [5, 7])

    def test_thresholds(self):
        curve = [(0, 0.1, 5), (1, 0.5, 5), (2, 0.92, 5), (4, 0.99, 5)]
        self.assertEqual(metrics.effective_fewshot_exposures(curve, 1.0, 0.9), 2)
        retention = [(1, 0.98, 5), (2, 0.9, 5), (4, 0.5, 5)]
        self.assertEqual(metrics.retention_time(retention, 0.98, 0.1), 4)


class TestModels(unittest.TestCase):

    def test_ltm_learns_and_reports_pre_update_prediction(self):
        torch.manual_seed(0)
        model = LTMOnly(MLP(4, [16], 2), lr=0.1, steps=3)
        x = torch.tensor([[1.0, 0, 0, 0]])
        y = torch.tensor([1])
        before = model.predict(x).argmax(1)
        pred = model.observe(x, y)
        self.assertEqual(pred.tolist(), before.tolist())
        for _ in range(20):
            model.observe(x, y)
        self.assertEqual(model.predict(x).argmax(1).item(), 1)

    def test_reservoir_capacity_and_memory(self):
        torch.manual_seed(0)
        model = ReplayLTM(MLP(4, [8], 2), lr=0.01, capacity=10, replay_batch=4, input_shape=(4,))
        for i in range(100):
            model.observe(torch.randn(1, 4), torch.tensor([i % 2]))
        self.assertEqual(model.size, 10)
        self.assertEqual(model.seen, 100)
        params = sum(p.numel() * 4 for p in model.net.parameters())
        self.assertEqual(model.memory_bytes(), params + 10 * 4 * 4 + 10 * 8)


    def test_continual_backprop_replacement(self):
        torch.manual_seed(0)
        net = MLP(6, [50, 50], 3)
        cbp = ContinualBackprop(net, rate=0.01, decay=0.9, maturity=10)
        model = LTMOnly(net, lr=0.01, cbp=cbp)
        x = torch.randn(64, 6)
        before = model.predict(x)
        for i in range(10):
            model.observe(x[i:i + 1], torch.tensor([i % 3]))
        self.assertEqual(cbp.replaced, 0)  # nothing is mature yet
        for i in range(400):
            model.observe(x[i % 64:i % 64 + 1], torch.tensor([i % 3]))
        # rate 0.01 x at most 100 eligible units (2 layers of 50) x ~390 mature steps: at most ~390, fewer
        # because replaced units are immature again for 10 steps
        self.assertGreater(cbp.replaced, 300)
        self.assertLessEqual(cbp.replaced, 400)
        self.assertTrue(torch.all(cbp.age[0] <= 410))

    def test_replacement_preserves_function_approximately(self):
        # replacing a unit zeroes its outgoing weights and folds its mean contribution into the next bias
        torch.manual_seed(0)
        net = MLP(4, [8], 2)
        cbp = ContinualBackprop(net, rate=1.0, decay=0.0, maturity=0)
        x = torch.randn(1, 4)
        y0 = net(x).detach()
        cbp.step()  # decay 0: utility = current contribution, mean = current activation; replaces 8 units
        y1 = net(x).detach()
        torch.testing.assert_close(y0, y1, atol=1e-5, rtol=0)


if __name__ == "__main__":
    unittest.main()
