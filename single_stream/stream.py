"""
Correlated single-stream generator (task 1 of single_stream_paper Notes/experiments/plan.md, section 2).

Each epoch: shuffle the samples within each group, cut every group into consecutive clusters of c samples (the
last one truncated), then repeatedly pick a non-exhausted group, with probability proportional to its remaining
clusters, and present its next cluster. When c > 1 the group just presented is excluded from the next pick
unless it is the only one left. When c = 1 there is no exclusion, so the epoch is an exact uniform shuffle.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class EpochOrder:

    order: np.ndarray            # sample indices in presentation order (each sample exactly once)
    cluster_groups: np.ndarray   # group id of each presented cluster, in order
    cluster_lengths: np.ndarray  # length of each presented cluster (c, or less when a group's tail is truncated)


class ClusteredStream:

    def __init__(self, groups, cluster_size, rng):
        if cluster_size < 1:
            raise ValueError(f"cluster_size must be >= 1, got {cluster_size}")
        self.groups = np.asarray(groups)
        self.cluster_size = int(cluster_size)
        self.rng = rng
        self.group_ids = np.unique(self.groups)
        self.members = [np.flatnonzero(self.groups == g) for g in self.group_ids]

    def epoch(self):
        c = self.cluster_size
        chunks = []
        for idx in self.members:
            shuffled = self.rng.permutation(idx)
            chunks.append([shuffled[i:i + c] for i in range(0, len(shuffled), c)])
        remaining = np.array([len(ch) for ch in chunks], dtype=np.int64)
        position = np.zeros(len(chunks), dtype=np.int64)

        order, cluster_groups, cluster_lengths = [], [], []
        previous = None
        while remaining.sum() > 0:
            weights = remaining.astype(np.float64)
            if c > 1 and previous is not None and np.count_nonzero(weights) > 1:
                weights[previous] = 0.0
            g = self.rng.choice(len(chunks), p=weights / weights.sum())
            cluster = chunks[g][position[g]]
            position[g] += 1
            remaining[g] -= 1
            previous = g
            order.append(cluster)
            cluster_groups.append(self.group_ids[g])
            cluster_lengths.append(len(cluster))

        return EpochOrder(np.concatenate(order), np.array(cluster_groups), np.array(cluster_lengths))


def run_lengths(sequence):
    """Lengths of the maximal runs of equal consecutive values."""
    sequence = np.asarray(sequence)
    if len(sequence) == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(sequence[1:] != sequence[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(sequence)]])
    return np.diff(bounds)
