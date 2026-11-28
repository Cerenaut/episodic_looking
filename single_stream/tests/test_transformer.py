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


class TestWhatWhere(unittest.TestCase):
    """The what+where bias (Dave 28 Nov 2026) and its coarse where grid."""

    def nets(self):
        torch.manual_seed(0)
        nets = {m: BiasableTransformer(num_tokens=64, token_size=8, num_classes=2, d_model=16, heads=2, bias_mode=m)
                for m in ("what", "where", "what+where")}
        state = nets["what"].state_dict()
        for n in nets.values():
            n.load_state_dict(state)
        return nets

    def test_parts_match_the_single_modes(self):
        nets = self.nets()
        x = torch.rand(3, 64 * 8)
        self.assertEqual(nets["what+where"].bias_size, 16 + 64)
        a_what, a_where = torch.randn(3, 16), torch.randn(3, 64)
        both = nets["what+where"]
        self.assertTrue(torch.equal(both(x, torch.zeros(3, 80)), both(x)))           # a = 0: exactly unbiased
        self.assertTrue(torch.allclose(both(x, torch.cat([a_what, torch.zeros(3, 64)], 1)), nets["what"](x, a_what)))
        self.assertTrue(torch.allclose(both(x, torch.cat([torch.zeros(3, 16), a_where], 1)), nets["where"](x, a_where)))
        combined = both(x, torch.cat([a_what, a_where], 1))
        self.assertFalse(torch.allclose(combined, nets["what"](x, a_what)))

    def test_coarse_basis(self):
        from single_stream.stm import coarse_basis
        nets = self.nets()
        self.assertIsNone(coarse_basis(nets["what+where"], 0))
        b = coarse_basis(nets["what+where"], 4)
        self.assertEqual(tuple(b.shape), (16 + 16, 16 + 64))
        self.assertTrue(torch.equal(b[:16, :16], torch.eye(16)))
        self.assertTrue(torch.equal(b[16:, 16:].sum(0), torch.ones(64)))          # each token in one coarse cell
        self.assertTrue(torch.equal(b[16:, 16:].sum(1), torch.full((16,), 4.0)))  # each cell covers 2 x 2 tokens
        self.assertEqual(tuple(coarse_basis(nets["where"], 2).shape), (4, 64))
        with self.assertRaises(ValueError):
            coarse_basis(nets["what"], 4)

    def test_rl_stm_with_coarse_what_where(self):
        import types
        from single_stream.run_stream import build_model, build_parser
        train = types.SimpleNamespace(x=torch.rand(4, 64 * 8), y=torch.zeros(4, dtype=torch.long), num_classes=2)
        args = build_parser().parse_args(["--model", "cls", "--net", "transformer", "--d-model", "16", "--heads", "2",
                                          "--bias-mode", "what+where", "--bias-grid", "4", "--stm-hidden", "64",
                                          "--stm-sparsity", "8", "--sleep-reset-stm", "--replay-period", "2",
                                          "--out", "unused"])
        model = build_model(args, train, token_shape=(64, 8))
        self.assertEqual(model.stm.model.bias_size, 32)             # the actor emits 16 what + 16 coarse where
        for i in range(3):                                           # through a sleep and an STM reset
            model.observe(torch.rand(1, 64 * 8), torch.tensor([i % 2]))
        self.assertGreater(model.resets, 0)
        self.assertEqual(tuple(model.stm.bias_basis.shape), (32, 80))
        res = model.stm.episode(torch.rand(2, 64 * 8))
        self.assertEqual(tuple(res["bias"].shape[1:]), (2, 32))


if __name__ == "__main__":
    unittest.main()


class TestPlasticityDiagnostics(unittest.TestCase):

    def test_ranges(self):
        from single_stream import plasticity
        torch.manual_seed(0)
        net = BiasableTransformer(num_tokens=49, token_size=32, num_classes=10, blocks=2)
        net.train()
        out = plasticity.diagnostics(net, torch.randn(64, 49 * 32))
        self.assertTrue(net.training)                                   # mode restored
        self.assertEqual(set(out), {"plast_dormant_b0", "plast_dormant_b1", "plast_eff_rank", "plast_weight_norm"})
        self.assertTrue(0 <= out["plast_dormant_b0"] <= 1)
        self.assertTrue(1 <= out["plast_eff_rank"] <= net.d_model)
        self.assertGreater(out["plast_weight_norm"], 0)
        self.assertAlmostEqual(plasticity.effective_rank(torch.eye(8)), 8.0, places=5)


class TestTransformerContinualBackprop(unittest.TestCase):

    def _setup(self, rate):
        from single_stream.models import TransformerContinualBackprop
        torch.manual_seed(0)
        net = BiasableTransformer(num_tokens=49, token_size=32, num_classes=10, blocks=2)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        cbp = TransformerContinualBackprop(net, rate=rate, maturity=5)
        cbp.opt = opt
        return net, opt, cbp

    def _train(self, net, opt, cbp, steps):
        x, y = torch.randn(8, 49 * 32), torch.randint(0, 10, (8,))
        for _ in range(steps):
            net.train()
            loss = torch.nn.functional.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            cbp.step()
        return x

    def test_replaces_units_and_moments(self):
        net, opt, cbp = self._setup(rate=0.05)
        x = self._train(net, opt, cbp, 20)
        self.assertGreater(cbp.replaced, 0)
        out = net.blocks[0].mlp[2]
        young = (cbp.age[0] < 20).nonzero().flatten()          # replaced at some step
        self.assertGreater(len(young), 0)
        just = (cbp.age[0] == 0).nonzero().flatten()
        if len(just):                                           # replaced on the last step: outgoing zero, moments zero
            self.assertTrue(torch.all(out.weight[:, just] == 0))
            self.assertTrue(torch.all(opt.state[out.weight]["exp_avg"][:, just] == 0))
            self.assertTrue(torch.all(opt.state[net.blocks[0].mlp[0].weight]["exp_avg_sq"][just] == 0))

    def test_function_preserved_at_replacement(self):
        net, opt, cbp = self._setup(rate=0.0)
        self._train(net, opt, cbp, 10)
        x = torch.randn(4, 49 * 32)
        net.eval()
        before = net(x).detach()
        cbp.rate = 0.2                                          # force replacements on one more step, no update
        n0 = cbp.replaced
        cbp.step()
        self.assertGreater(cbp.replaced, n0)
        after = net(x).detach()
        # outgoing weights zeroed and the mean contribution moved to the bias: outputs change only by the
        # units' deviation from their running mean, so they stay close
        self.assertLess((after - before).abs().max().item(), 1.0)

    def test_eval_does_not_record_and_pickles(self):
        import io
        net, opt, cbp = self._setup(rate=0.0)
        net.eval()
        net(torch.randn(2, 49 * 32))
        self.assertTrue(all(a is None for a in cbp.acts))
        buf = io.BytesIO()
        torch.save(net, buf)

    def test_resumed_recorder_survives_diagnostics(self):
        """A snapshot loaded in a new process (hook counter back at 0): the diagnostics' hook must not displace the
        CBP recorder (plast12 resume bug, 26 Nov 2026); run_tasks.advance_hook_ids prevents it."""
        import io
        from torch.utils.hooks import RemovableHandle
        from single_stream import plasticity
        from single_stream.models import _ActRecorder
        from single_stream.run_tasks import advance_hook_ids
        net, opt, cbp = self._setup(rate=0.0)
        buf = io.BytesIO()
        torch.save({"net": net, "cbp": cbp}, buf)
        saved = RemovableHandle.next_id
        x = torch.randn(3, 49 * 32)
        try:
            for fix in (False, True):
                buf.seek(0)
                snap = torch.load(buf, weights_only=False)
                RemovableHandle.next_id = min(snap["net"].blocks[0].mlp[1]._forward_hooks)  # as in a new process
                if fix:
                    advance_hook_ids()
                plasticity.diagnostics(snap["net"], x)
                recorders = [h for b in snap["net"].blocks for h in b.mlp[1]._forward_hooks.values()
                             if isinstance(h, _ActRecorder)]
                self.assertEqual(len(recorders) == 2, fix)  # without the fix the diagnostics take the recorders' ids
                snap["net"].train()
                snap["net"](x)
                self.assertEqual(all(a is not None for a in snap["cbp"].acts), fix)
        finally:
            RemovableHandle.next_id = max(RemovableHandle.next_id, saved)

    def test_rate_zero_replaces_nothing(self):
        net, opt, cbp = self._setup(rate=0.0)
        torch.manual_seed(1)
        self._train(net, opt, cbp, 5)
        self.assertEqual(cbp.replaced, 0)

    def test_no_fold_leaves_bias(self):
        from single_stream.models import TransformerContinualBackprop
        torch.manual_seed(0)
        net = BiasableTransformer(num_tokens=49, token_size=32, num_classes=10, blocks=1)
        cbp = TransformerContinualBackprop(net, rate=0.0, maturity=5, fold=False)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)          # lr 0: only continual backprop changes weights
        cbp.opt = opt
        self._train(net, opt, cbp, 10)
        bias = net.blocks[0].mlp[2].bias.clone()
        cbp.rate = 0.2
        cbp.step()
        self.assertGreater(cbp.replaced, 0)
        self.assertTrue(torch.equal(net.blocks[0].mlp[2].bias, bias))
