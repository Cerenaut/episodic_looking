import unittest

import torch

from single_stream.preprocessor import ConvPreprocessor, PretrainNet, augment_rgb


class TestConvPreprocessor(unittest.TestCase):

    def test_mnist_layout_unchanged(self):
        """The defaults keep the MNIST layout (three convs + 1x1), so existing checkpoints load as they are."""
        pre = ConvPreprocessor(32, 64)
        self.assertEqual(pre.num_tokens, 49)
        self.assertEqual([tuple(m.weight.shape) for m in pre.body if isinstance(m, torch.nn.Conv2d)],
                         [(32, 1, 3, 3), (64, 32, 3, 3), (64, 64, 3, 3), (32, 64, 1, 1)])
        self.assertEqual(pre(torch.rand(2, 784)).shape, (2, 49, 32))

    def test_cifar_layout(self):
        pre = ConvPreprocessor(32, 64, in_channels=3, size=32, stem=False)
        self.assertEqual(pre.num_tokens, 64)
        self.assertEqual([tuple(m.weight.shape) for m in pre.body if isinstance(m, torch.nn.Conv2d)],
                         [(32, 3, 3, 3), (64, 32, 3, 3), (32, 64, 1, 1)])
        self.assertEqual(pre(torch.rand(2, 3072)).shape, (2, 64, 32))
        self.assertEqual(PretrainNet(pre, 200)(torch.rand(2, 3072)).shape, (2, 200))

    def test_augment_rgb_crops_and_flips(self):
        x = torch.rand(64, 3072)
        out = augment_rgb(x, torch.Generator().manual_seed(0))
        self.assertEqual(out.shape, x.shape)
        img, aug = x.reshape(-1, 3, 32, 32), out.reshape(-1, 3, 32, 32)
        # a zero shift without a flip, or a flip of the same crop, leaves the centre region equal to the original
        centre = lambda a: a[:, 8:24, 8:24]  # one (3, 32, 32) image
        same = [(centre(aug[i]) == centre(img[i])).all() or (centre(aug[i]) == centre(img[i].flip(2))).all()
                for i in range(64)]
        self.assertTrue(any(same) and not all(same))


if __name__ == "__main__":
    unittest.main()
