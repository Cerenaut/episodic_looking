"""
Dataset sequences for lifelong learning on a single stream (Dave, 5 Nov 2026): a task is the base dataset under a
transform, and the stream switches instantly from one task to the next, unannounced to the model. Transforms keep the
frozen preprocessor's features meaningful (no pixel permutation):

  dihedral image transforms on the 28 x 28 image: id, rot90, rot180, rot270 (counter-clockwise), hflip (mirror
  left-right), vflip (upside down), transpose, antitranspose;
  label permutations: lperm<k>, the digit labels relabelled by a fixed permutation drawn with seed 1000 + k (never the
  identity); the images are unchanged.

A task's samples are featurised after the transform (the frozen conv net sees the transformed image). Groups (the unit
of clustering) are the task's labels, so a label permutation clusters the same images by their new labels.
"""
import numpy as np
import torch

from single_stream.datasets import StreamDataset

DIHEDRAL = {
    "id": lambda im: im,
    "rot90": lambda im: torch.rot90(im, 1, (1, 2)),
    "rot180": lambda im: torch.rot90(im, 2, (1, 2)),
    "rot270": lambda im: torch.rot90(im, 3, (1, 2)),
    "hflip": lambda im: torch.flip(im, (2,)),
    "vflip": lambda im: torch.flip(im, (1,)),
    "transpose": lambda im: im.transpose(1, 2),
    "antitranspose": lambda im: torch.rot90(im, 2, (1, 2)).transpose(1, 2),
}


def label_permutation(k, num_classes=10):
    """The fixed label permutation of lperm<k> (seed 1000 + k), redrawn until it is not the identity."""
    rng = np.random.default_rng(1000 + k)
    while True:
        p = rng.permutation(num_classes)
        if (p != np.arange(num_classes)).any():
            return torch.from_numpy(p)


def valid(name):
    return name in DIHEDRAL or (name.startswith("lperm") and name[5:].isdigit())


def apply(name, data):
    """The task `name` on raw (N, 784) images: a new StreamDataset (x still raw pixels)."""
    if not valid(name):
        raise ValueError(f"unknown task {name}")
    if name in DIHEDRAL:
        side = int(round(data.x.shape[1] ** 0.5))
        x = DIHEDRAL[name](data.x.reshape(-1, side, side)).reshape(len(data.x), -1).contiguous()
        return StreamDataset(x, data.y, data.groups, data.num_classes, f"{data.name}:{name}")
    perm = label_permutation(int(name[5:]), data.num_classes)
    y = perm[data.y]
    return StreamDataset(data.x, y, y.numpy().copy(), data.num_classes, f"{data.name}:{name}")


def image_transform(name):
    """The image part of a task: its dihedral transform, or id for a label permutation."""
    return name if name in DIHEDRAL else "id"


def featurized(name, data, pre, cache=None):
    """The task's dataset with the frozen preprocessor's tokens as x (pre = None: raw pixels). `cache` (a directory):
    tokens are stored per image transform and split, so label permutations reuse the original images' tokens and
    runs share them; the key includes the data's size and a checksum of its labels, and the preprocessor's tag."""
    import os
    from single_stream.preprocessor import featurize
    d = apply(name, data)
    if pre is None:
        return d
    path = None
    if cache:
        tag = getattr(pre, "tag", "pre")
        path = os.path.join(cache, f"{tag}_{image_transform(name)}_{len(data.y)}_{int(data.y.sum())}.pt")
        if os.path.exists(path):
            return StreamDataset(torch.load(path, weights_only=True), d.y, d.groups, d.num_classes, d.name)
    with torch.no_grad():
        x = featurize(pre, apply(image_transform(name), data).x)
    if path:
        os.makedirs(cache, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(x, tmp)
        os.replace(tmp, path)
    return StreamDataset(x, d.y, d.groups, d.num_classes, d.name)
