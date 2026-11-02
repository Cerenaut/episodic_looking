import math
import unittest

import torch

from model.transformer import BiasableTransformer


class TestBiasableTransformer(unittest.TestCase):

    def make(self, mode, blocks=1, bias_block=0):
        torch.manual_seed(0)
        return BiasableTransformer(num_tokens=49, token_size=32, num_classes=10, blocks=blocks, bias_block=bias_block,
                                   bias_mode=mode).eval()

    def test_neutral_bias_is_identity(self):
        x = torch.rand(5, 49 * 32)
        for mode in ("where", "what"):
            net = self.make(mode)
            a = net.run(x)
            b = net.run(x, torch.zeros(5, net.bias_size))
            for key in ("features", "logits"):
                torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=0)

    def test_bias_changes_output_not_pre_bias_state(self):
        x = torch.rand(5, 49 * 32)
        for mode in ("where", "what"):
            net = self.make(mode)
            a = net.run(x)
            b = net.run(x, torch.randn(5, net.bias_size) * 3)
            torch.testing.assert_close(a["pre_bias"], b["pre_bias"], atol=0, rtol=0)
            self.assertGreater((a["logits"] - b["logits"]).abs().max().item(), 1e-3)

    def test_bias_sizes(self):
        self.assertEqual(self.make("where").bias_size, 49)
        self.assertEqual(self.make("what").bias_size, 64)

    def test_deeper_bias_block(self):
        net = self.make("where", blocks=3, bias_block=2)
        x = torch.rand(2, 49 * 32)
        a = net.run(x)
        b = net.run(x, torch.randn(2, 49))
        torch.testing.assert_close(a["pre_bias"], b["pre_bias"], atol=0, rtol=0)  # earlier blocks unaffected

    def test_where_bias_is_bounded(self):
        """Saturated biases reweight attention by at most e^(2s) between any two keys: the result equals that of a
        bias of exactly +-s, so no token can be suppressed completely."""
        net = self.make("where")
        x = torch.rand(3, 49 * 32)
        a = torch.where(torch.rand(3, 49) > 0.5, 1e6, -1e6)
        s = net.bias_scale
        sat = net.run(x, a)["logits"]
        bound = net.run(x, torch.sign(a) * 1e3)["logits"]
        torch.testing.assert_close(sat, bound)
        self.assertAlmostEqual(s, math.log(4.0))


class TestQKVGains(unittest.TestCase):
    """Per-channel gains on the projected queries / keys / values (Dave, 2 Nov 2026)."""

    def make(self, mode):
        torch.manual_seed(0)
        net = BiasableTransformer(num_tokens=49, token_size=8, num_classes=10, d_model=16, heads=2, bias_mode=mode)
        return net.eval()

    def test_sizes_and_neutral_at_zero(self):
        x = torch.randn(3, 49 * 8)
        for mode, parts in (("q", 1), ("kv", 2), ("qkv", 3)):
            net = self.make(mode)
            self.assertEqual(net.bias_size, 16 * parts)
            torch.testing.assert_close(net(x, torch.zeros(3, net.bias_size)), net(x))
            self.assertFalse(torch.allclose(net(x, torch.randn(3, net.bias_size)), net(x)))

    def test_q_and_k_gains_are_equivalent_with_one_block(self):
        # only the CLS row is read out; its score for key j is sum_c q_c g_c k_jc either way
        x, a = torch.randn(3, 49 * 8), torch.randn(3, 16)
        torch.testing.assert_close(self.make("q")(x, a), self.make("k")(x, a))


if __name__ == "__main__":
    unittest.main()
