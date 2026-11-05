"""Unit tests of the task sequence transforms (single_stream/tasks.py)."""
import unittest

import numpy as np
import torch

from single_stream import tasks
from single_stream.datasets import StreamDataset


def one_pixel(row, col, side=28):
    x = torch.zeros(1, side * side)
    x[0, row * side + col] = 1.0
    return StreamDataset(x, torch.tensor([3]), np.array([3]), 10, "t")


def where(d, side=28):
    i = int(d.x[0].argmax())
    return divmod(i, side)


class TestDihedral(unittest.TestCase):

    def test_pixel_moves(self):
        d = one_pixel(2, 5)   # row 2, column 5 (upper area, left of centre)
        expect = {"id": (2, 5), "hflip": (2, 22), "vflip": (25, 5), "rot180": (25, 22), "transpose": (5, 2),
                  "rot90": (22, 2),          # counter-clockwise: the top row moves to the left column
                  "rot270": (5, 25), "antitranspose": (22, 25)}
        for name, rc in expect.items():
            self.assertEqual(where(tasks.apply(name, d)), rc, name)

    def test_group_of_eight(self):
        x = torch.rand(3, 784)
        d = StreamDataset(x, torch.arange(3), np.arange(3), 10, "t")
        outs = {name: tasks.apply(name, d).x for name in tasks.DIHEDRAL}
        names = list(outs)
        for i, a in enumerate(names):   # all eight are distinct on a random image
            for b in names[i + 1:]:
                self.assertFalse(torch.equal(outs[a], outs[b]), (a, b))
        twice = tasks.apply("rot90", tasks.apply("rot90", d)).x
        torch.testing.assert_close(twice, outs["rot180"])
        torch.testing.assert_close(tasks.apply("hflip", tasks.apply("hflip", d)).x, x)

    def test_labels_unchanged(self):
        d = one_pixel(2, 5)
        out = tasks.apply("rot90", d)
        self.assertEqual(int(out.y[0]), 3)
        self.assertEqual(int(out.groups[0]), 3)


class TestLabelPermutation(unittest.TestCase):

    def test_fixed_bijective_not_identity(self):
        for k in range(1, 8):
            p = tasks.label_permutation(k)
            self.assertEqual(sorted(p.tolist()), list(range(10)))
            self.assertFalse(torch.equal(p, torch.arange(10)))
            torch.testing.assert_close(p, tasks.label_permutation(k))
        self.assertFalse(torch.equal(tasks.label_permutation(1), tasks.label_permutation(2)))

    def test_images_same_labels_and_groups_permuted(self):
        x = torch.rand(10, 784)
        d = StreamDataset(x, torch.arange(10), np.arange(10), 10, "t")
        out = tasks.apply("lperm1", d)
        torch.testing.assert_close(out.x, x)
        p = tasks.label_permutation(1)
        torch.testing.assert_close(out.y, p)
        np.testing.assert_array_equal(out.groups, p.numpy())

    def test_names(self):
        self.assertTrue(tasks.valid("lperm3"))
        self.assertFalse(tasks.valid("lperm"))
        self.assertFalse(tasks.valid("rot45"))


if __name__ == "__main__":
    unittest.main()
