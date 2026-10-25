"""Run: python -m unittest discover -s single_stream/tests -t ."""
import itertools
import unittest
from collections import Counter

import numpy as np
import torch

from single_stream import metrics
from single_stream.buffers import NeuralBuffer, NoveltyGate, RingBuffer, part_specs
from single_stream.models import BufferedReplayLTM, ContinualBackprop, LTMOnly, MLP, ReplayLTM, ReplayOnlyLTM
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


    def test_replay_only_fifo(self):
        torch.manual_seed(0)
        model = ReplayOnlyLTM(MLP(4, [8], 3), lr=0.01, capacity=5, batch=2, period=3, steps=2, input_shape=(4,))
        xs = torch.randint(0, 256, (20, 4)).float() / 255.0
        for i in range(20):
            model.observe(xs[i:i + 1], torch.tensor([i % 3]))
        self.assertEqual(model.size, 5)
        self.assertEqual(model.updates, 2 * (20 // 3))
        # the buffer holds exactly the last 5 exposures, stored exactly as 8-bit
        held = sorted(model._decode(model.buf_x).tolist())
        self.assertEqual(held, sorted(xs[15:].tolist()))
        self.assertEqual(sorted(model.buf_y.tolist()), sorted([i % 3 for i in range(15, 20)]))
        params = sum(p.numel() * 4 for p in model.net.parameters())
        self.assertEqual(model.memory_bytes(), params + 5 * 4 * 1 + 5 * 8)

    def test_replay_only_does_not_train_between_updates(self):
        torch.manual_seed(0)
        model = ReplayOnlyLTM(MLP(4, [8], 3), lr=0.1, capacity=10, batch=2, period=4, input_shape=(4,))
        w0 = model.net.out.weight.clone()
        for i in range(3):
            model.observe(torch.rand(1, 4), torch.tensor([0]))
        torch.testing.assert_close(model.net.out.weight, w0)
        model.observe(torch.rand(1, 4), torch.tensor([0]))
        self.assertFalse(torch.equal(model.net.out.weight, w0))


class TestBuffers(unittest.TestCase):

    @staticmethod
    def offer(buf, xs, ys, novelty=None):
        for i in range(len(xs)):
            nov = torch.ones(1) if novelty is None else novelty[i:i + 1]
            buf.write((xs[i:i + 1], ys[i:i + 1]), nov)

    @staticmethod
    def image_label_parts(d):
        return part_specs(torch.zeros(1, d), torch.zeros(1, dtype=torch.long))

    def test_pack_unpack_parts(self):
        parts = (torch.rand(3, 2, 4), torch.tensor([5, 0, 9]), torch.rand(3, 1))
        buf = RingBuffer(10, part_specs(*parts))
        self.assertEqual(buf.D, 8 + 1 + 1)
        buf.write(parts, torch.ones(3))
        out = buf.unpack(buf.sample_vectors(3))
        self.assertEqual([o.shape for o in out], [(3, 2, 4), (3,), (3, 1)])
        self.assertEqual(out[1].dtype, torch.long)
        # sampling without replacement returns each stored sample once
        got = sorted(out[1].tolist())
        self.assertEqual(got, [0, 5, 9])
        for k in range(3):
            row = out[1].tolist().index(parts[1][k].item())
            torch.testing.assert_close(out[0][row], parts[0][k])
            torch.testing.assert_close(out[2][row], parts[2][k])

    def test_ring_buffer_model_reproduces_replay_only(self):
        xs = torch.randint(0, 256, (200, 4)).float() / 255.0
        ys = torch.arange(200) % 3
        torch.manual_seed(0)
        a = ReplayOnlyLTM(MLP(4, [8], 3), lr=0.05, capacity=20, batch=4, period=3, steps=2, input_shape=(4,), seed=7)
        torch.manual_seed(0)
        b = BufferedReplayLTM(MLP(4, [8], 3), lr=0.05, buffer=RingBuffer(20, self.image_label_parts(4), seed=7),
                              batch=4, period=3, steps=2)
        for i in range(200):
            pa = a.observe(xs[i:i + 1], ys[i:i + 1])
            pb = b.observe(xs[i:i + 1], ys[i:i + 1])
            self.assertEqual(pa.tolist(), pb.tolist())
        for wa, wb in zip(a.net.parameters(), b.net.parameters()):
            torch.testing.assert_close(wa, wb, rtol=0, atol=0)
        self.assertEqual(b.memory_values(), a.memory_values())
        self.assertEqual(b.memory_values(), sum(p.numel() for p in b.net.parameters()) + 20 * 5)

    def test_gate(self):
        gate = NoveltyGate(floor=0.0, seed=0)
        self.assertFalse(gate(torch.zeros(1000)).any())
        self.assertTrue(gate(torch.ones(1000)).all())
        self.assertAlmostEqual(NoveltyGate(floor=0.3, seed=0)(torch.zeros(10000)).float().mean().item(), 0.3, delta=0.02)
        buf = RingBuffer(10, self.image_label_parts(4), gate=NoveltyGate(floor=0.0))
        self.offer(buf, torch.rand(50, 4), torch.zeros(50, dtype=torch.long), torch.zeros(50))
        self.assertEqual((buf.offered, buf.written, buf.size), (50, 0, 0))

    def test_neural_buffer_is_fifo_with_exact_recall(self):
        xs, ys = torch.rand(25, 20), torch.arange(25) % 4
        buf = NeuralBuffer(8, self.image_label_parts(20), fan_in=5, key_parts=(0,), util_decay=0.5, seed=0)
        self.offer(buf, xs, ys)
        x, y = buf.sample(8)  # without replacement: every memory once
        held = sorted(zip(x.tolist(), y.tolist()))
        expected = sorted(zip(xs[17:].tolist(), ys[17:].tolist()))
        self.assertEqual(held, expected)
        self.assertEqual(buf.num_values(), 8 * 21 + 2 * 8)

    def test_fifo_order_survives_utilisation_underflow(self):
        buf = NeuralBuffer(4, self.image_label_parts(10), fan_in=3, util_decay=1e-30, seed=0)
        xs = torch.rand(10, 10)
        self.offer(buf, xs, torch.zeros(10, dtype=torch.long))
        x, _ = buf.sample(4)
        self.assertEqual(sorted(x.tolist()), sorted(xs[6:].tolist()))

    def test_key_excludes_label(self):
        buf = NeuralBuffer(50, self.image_label_parts(20), fan_in=20, key_parts=(0,), seed=0)
        self.assertTrue(torch.all(buf.idx < 20))

    def test_recognition_skips_duplicates(self):
        buf = NeuralBuffer(10, self.image_label_parts(30), fan_in=10, key_parts=(0,), match_threshold=0.99, seed=0)
        x = torch.rand(1, 30)
        for _ in range(3):
            buf.write((x, torch.zeros(1, dtype=torch.long)), torch.ones(1))
        self.assertEqual((buf.offered, buf.written, buf.recognised), (3, 1, 2))
        self.assertGreater(buf.util.max().item(), 2.0)
        # a rejected, unrecognised sample changes nothing
        buf.gate = NoveltyGate(floor=0.0)
        c0 = buf.C.clone()
        buf.write((torch.rand(1, 30), torch.zeros(1, dtype=torch.long)), torch.zeros(1))
        self.assertTrue(torch.equal(buf.C, c0))


if __name__ == "__main__":
    unittest.main()
