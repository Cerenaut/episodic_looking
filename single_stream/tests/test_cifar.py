import unittest

import numpy as np

from single_stream import cifar


def fake_cifar(per_class=500):
    """Labels shaped like CIFAR-100's training file: 20 coarse classes of 5 fine classes each."""
    coarse_of = np.repeat(np.arange(20), 5)
    rng = np.random.default_rng(0)
    perm = rng.permutation(100)           # fine ids not contiguous within a coarse class, as in CIFAR-100
    fine = np.repeat(perm, per_class)
    coarse = coarse_of[np.argsort(perm)][fine]
    return fine, coarse


class TestSchedule(unittest.TestCase):

    def setUp(self):
        self.fine, self.coarse = fake_cifar()
        self.train, self.val = cifar.validation_split(self.fine, 100)

    def make(self, seed=0, c=64, epochs=2):
        return cifar.make_schedule((3, 7), self.fine, self.coarse, self.train, seed, c, epochs)

    def test_classes_and_phases(self):
        s = self.make()
        a, b = cifar.fine_of(self.coarse, self.fine, 3), cifar.fine_of(self.coarse, self.fine, 7)
        self.assertEqual(sorted(s["order"]["a"]), a)
        self.assertEqual(sorted(s["order"]["b"]), b)
        self.assertEqual(s["pretrain"], [s["order"]["a"][0], s["order"]["b"][0]])
        intros = s["introductions"]
        self.assertEqual(len(intros), 8)
        self.assertTrue(all(f in a for f in intros[0::2]) and all(f in b for f in intros[1::2]))  # alternating sides
        kinds = [p["kind"] for p in s["phases"]]
        self.assertEqual(kinds, ["pretrain"] + ["intro"] * 8 + ["resilience"])
        self.assertEqual(s["phases"][-1]["new"], intros[0])
        self.assertEqual(sum(p["length"] for p in s["phases"]), len(s["stream"]))

    def test_phase_contents(self):
        for c in (32, 64, 256):
            s = self.make(c=c)
            stream = s["stream"]
            self.assertFalse(np.isin(stream, self.val).any())        # never a validation image
            pre = s["phases"][0]
            seg = stream[pre["start"]:pre["start"] + pre["length"]]
            self.assertEqual(set(self.fine[seg]), set(s["pretrain"]))
            self.assertEqual(len(seg), 2 * 800)                      # 2 epochs of 2 classes x 400 images
            seen = set(s["pretrain"])
            for ph in s["phases"][1:]:
                seg = stream[ph["start"]:ph["start"] + ph["length"]]
                new = self.fine[seg] == ph["new"]
                self.assertEqual(new.sum(), 400)                     # every training image of the new class once
                self.assertEqual(len(set(seg[new])), 400)
                self.assertAlmostEqual(new.mean(), 0.75, places=2)
                self.assertEqual(len(set(seg)), len(seg))            # no image twice in a phase
                self.assertTrue(set(self.fine[seg[~new]]) <= seen - {ph["new"]})
                seen.add(ph["new"])

    def test_others_interleaved(self):
        from single_stream.stream import run_lengths
        s = self.make(c=32)
        for ph in s["phases"][1:]:
            seg = s["stream"][ph["start"]:ph["start"] + ph["length"]]
            others = self.fine[seg] != ph["new"]
            # the other classes' clusters are interleaved with the new class's, not all in one block
            runs = run_lengths(np.where(others, self.fine[seg], -1))
            self.assertGreater(len(runs), 1)

    def test_seed_determinism_and_hash(self):
        a, b, c = self.make(seed=1), self.make(seed=1), self.make(seed=2)
        self.assertTrue(np.array_equal(a["stream"], b["stream"]))
        self.assertEqual(cifar.stream_hash(a["stream"]), cifar.stream_hash(b["stream"]))
        self.assertNotEqual(cifar.stream_hash(a["stream"]), cifar.stream_hash(c["stream"]))
        # the validation split depends on split_seed only
        self.assertTrue(np.array_equal(cifar.validation_split(self.fine, 100)[1], self.val))

    def test_paired(self):
        from single_stream.stream import run_lengths
        for c in (32, 64, 256):
            s = cifar.make_schedule((3, 7), self.fine, self.coarse, self.train, 0, c, 2, paired=True)
            a, b = s["order"]["a"], s["order"]["b"]
            self.assertEqual(s["pairs"], [[a[k], b[k]] for k in range(1, 5)])
            self.assertEqual([p["kind"] for p in s["phases"]], ["pretrain"] + ["intro"] * 4 + ["resilience"])
            self.assertEqual(s["phases"][-1]["new"], s["pairs"][0])
            self.assertEqual(sum(p["length"] for p in s["phases"]), len(s["stream"]))
            for ph in s["phases"][1:]:
                seg = s["stream"][ph["start"]:ph["start"] + ph["length"]]
                self.assertFalse(np.isin(seg, self.val).any())
                self.assertEqual(len(set(seg)), len(seg))
                self.assertEqual(len(seg), 800)                      # both classes' 400 images, nothing else
                self.assertEqual(set(self.fine[seg]), set(ph["new"]))
                # alternating clusters of c: a, b, a, b, ...
                runs = run_lengths(self.fine[seg])
                self.assertTrue(all(r == c for r in runs[:2 * (400 // c)]))
                starts = self.fine[seg][np.concatenate([[0], np.cumsum(runs)[:-1]])]
                self.assertTrue((starts[1:] != starts[:-1]).all())
        # the single-class mode is unchanged by the option
        self.assertTrue(np.array_equal(self.make()["stream"], cifar.make_schedule(
            (3, 7), self.fine, self.coarse, self.train, 0, 64, 2, paired=False)["stream"]))


class TestScore(unittest.TestCase):

    def test_metrics_from_records(self):
        phases = [{"kind": "pretrain", "new": None}, {"kind": "intro", "new": 5}, {"kind": "resilience", "new": 5}]

        def rec(phase, n, acc, mem=None, seen=(1, 2)):
            return {"phase": phase, "n": n, "mem": mem, "seen": list(seen), "acc": {str(k): v for k, v in acc.items()}}

        recs = [rec(0, 0, {1: .9, 2: .8, 5: .2}),
                rec(1, 0, {1: .9, 2: .8, 5: .2}),
                rec(1, 1, {1: .7, 2: .8, 5: .4}, mem=1.0),
                rec(1, 2, {1: .9, 2: .9, 5: .6}, mem=.5),
                rec(1, 2, {1: .1, 2: .1, 5: .9}, mem=.5),   # a later checkpoint at the same n: Protection only
                rec(2, 0, {1: .9, 2: .9, 5: .5}, seen=(1, 2, 5)),
                rec(2, 1, {1: .9, 2: .9, 5: .7}, mem=1.0, seen=(1, 2, 5)),
                rec(2, 2, {1: .9, 2: .9, 5: .9}, mem=1.0, seen=(1, 2, 5))]
        m = cifar.score(recs, phases, points=(1, 2))
        self.assertAlmostEqual(m["Memorisation"], .75)
        self.assertAlmostEqual(m["Generalisation"], .5)        # the first record at n = 1, 2: .4, .6
        self.assertAlmostEqual(m["zero_shot"], .2)
        self.assertNotIn("gain", m)
        self.assertAlmostEqual(m["Protection"], .1)            # worst pooled accuracy on classes 1, 2
        self.assertAlmostEqual(m["Retention"], (.1 + .1 + .9) / 3)  # the phase's last checkpoint, seen + new
        self.assertAlmostEqual(m["Resilience"], .8)
        core = [.75, .5, .1, (.1 + .1 + .9) / 3, .8]
        self.assertAlmostEqual(m["Overall"], np.mean(core))
        self.assertAlmostEqual(m["Overall - worst"], .1)
        self.assertAlmostEqual(m["rescaled"]["Overall - worst"], -.8)

    def test_cut_stream_scores_without_resilience(self):
        phases = [{"kind": "pretrain", "new": None}, {"kind": "intro", "new": 5}, {"kind": "resilience", "new": 5}]
        recs = [{"phase": 1, "n": n, "mem": m, "seen": [1, 2], "acc": {"1": .9, "2": .8, "5": a}}
                for n, m, a in ((0, None, .2), (1, 1.0, .4), (2, .5, .6))]
        m = cifar.score(recs, phases, points=(1, 2))
        self.assertTrue(np.isnan(m["Resilience"]))
        self.assertEqual(m["core"], ["Memorisation", "Generalisation", "Protection", "Retention"])
        self.assertAlmostEqual(m["Overall"], np.mean([.75, .5, .85, (.9 + .8 + .6) / 3]))

    def test_paired_constant_predictor_scores_half(self):
        # pair (5, 6) on opposite sides; seen 1 (side of 5), 2 (side of 6). A model answering the current cluster's
        # label: right on the current class and its side, wrong on the other
        phases = [{"kind": "pretrain", "new": None}, {"kind": "intro", "new": [5, 6]},
                  {"kind": "resilience", "new": [5, 6]}]
        side = {1: 0, 5: 0, 2: 1, 6: 1}

        def rec(phase, n5, n6, current, seen=(1, 2)):
            acc = {str(f): float(side[f] == side[current]) for f in side}
            mem = {"5": 1.0 if current == 5 and n5 else (0.0 if n5 else None),
                   "6": 1.0 if current == 6 and n6 else (0.0 if n6 else None)}
            return {"phase": phase, "n": {"5": n5, "6": n6}, "mem": mem, "seen": list(seen), "acc": acc}

        recs = [rec(1, 0, 0, 1), rec(1, 1, 0, 5), rec(1, 2, 0, 5), rec(1, 2, 1, 6), rec(1, 2, 2, 6),
                rec(2, 0, 0, 6, seen=(1, 2, 5, 6)), rec(2, 1, 0, 5, seen=(1, 2, 5, 6)),
                rec(2, 2, 0, 5, seen=(1, 2, 5, 6)), rec(2, 2, 1, 6, seen=(1, 2, 5, 6)),
                rec(2, 2, 2, 6, seen=(1, 2, 5, 6))]
        m = cifar.score(recs, phases, points=(1, 2))
        for k in ("Memorisation", "Generalisation", "Protection", "Retention", "Resilience"):
            self.assertAlmostEqual(m[k], .5, msg=k)
        self.assertAlmostEqual(m["Overall"], .5)


if __name__ == "__main__":
    unittest.main()
