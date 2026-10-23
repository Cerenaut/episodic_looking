"""
Image-classification datasets for the single-stream tasks, behind one generic container: samples, target labels
and a grouping key (the unit of clustering). Add a loader to LOADERS to support a new dataset.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StreamDataset:

    x: torch.Tensor       # float32, (N, ...) model input
    y: torch.Tensor       # int64, (N,) target labels
    groups: np.ndarray    # int, (N,) grouping key for clustering and per-group metrics
    num_classes: int
    name: str = ""

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return StreamDataset(self.x[idx], self.y[idx], self.groups[idx], self.num_classes, self.name)


def stratified_holdout(groups, per_group, rng):
    """Split indices into (train, holdout) with `per_group` holdout samples from every group."""
    holdout = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        holdout.append(rng.choice(idx, size=per_group, replace=False))
    holdout = np.sort(np.concatenate(holdout))
    train = np.setdiff1d(np.arange(len(groups)), holdout)
    return train, holdout


def permute_pixels(x, permutation):
    """Apply a pixel permutation to flattened inputs (permuted MNIST)."""
    return x[:, permutation]


def load_mnist(root, permutation_seed=None):
    """MNIST as flattened 784-d inputs scaled to [0, 1]; groups = digit classes. Returns (train, test)."""
    from torchvision.datasets import MNIST

    out = []
    for train in (True, False):
        d = MNIST(root, train=train, download=False)
        x = d.data.reshape(len(d.data), -1).float() / 255.0
        y = d.targets.long()
        out.append(StreamDataset(x, y, y.numpy().copy(), 10, "mnist"))
    if permutation_seed is not None:
        perm = torch.from_numpy(np.random.default_rng(permutation_seed).permutation(784))
        out = [StreamDataset(permute_pixels(d.x, perm), d.y, d.groups, d.num_classes, "pmnist") for d in out]
    return tuple(out)


LOADERS = {"mnist": load_mnist}


def load(name, root, val_holdout=0, split_seed=0, **kwargs):
    """
    Returns (train, eval_set). With val_holdout > 0, `val_holdout` samples per group are cut from the training set
    and returned as the evaluation set (selection runs); otherwise the full training set and the test set.
    """
    train, test = LOADERS[name](root, **kwargs)
    if val_holdout > 0:
        tr, va = stratified_holdout(train.groups, val_holdout, np.random.default_rng(split_seed))
        return train.subset(tr), train.subset(va)
    return train, test
