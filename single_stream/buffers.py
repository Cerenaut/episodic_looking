"""
Replay buffers for replay-trained models.

A sample is a sequence of parts, e.g. (image, label); a buffer stores each sample as one vector, its parts
flattened and concatenated, and splits it back into parts (original shapes and dtypes) when sampling. Buffers know
nothing about what the parts mean. Whether a sample is worth storing is decided from a novelty score in [0, 1]
supplied by the caller (it is a function of the LTM: in task 1, the LTM's error on the sample, computed by the
model); the buffer's gate turns it into a stochastic store decision. Buffers never receive group, cluster or
task-switch information.

Shapes used below:
  n: number of samples in one call: samples offered by `write` (the stream's minibatch b) or drawn by `sample`
     (the replay batch).
  D: number of values in one stored sample, summed over its parts (MNIST: 784 pixels + 1 label = 785).

Memory is counted in values (plan.md section 4): one per stored number whatever its precision, counting every
value whose content depends on the data seen; fixed, seed-generated values are not counted.
"""
import math

import torch


class NoveltyGate:
    """Store with probability floor + (1 - floor) x novelty. floor = 1 stores everything and draws no random numbers."""

    def __init__(self, floor=1.0, seed=0):
        self.floor = floor
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, novelty):
        if self.floor >= 1:
            return torch.ones(len(novelty), dtype=torch.bool)
        p = self.floor + (1 - self.floor) * novelty.clamp(0, 1)
        return torch.rand(len(novelty), generator=self.gen) < p


def part_specs(*parts):
    """Per-sample (shape, dtype) of each part, from a batch of parts each shaped (n, ...)."""
    return [(tuple(p.shape[1:]), p.dtype) for p in parts]


class Buffer:
    """
    parts: per-sample (shape, dtype) of each part (see `part_specs`). Subclasses implement `_offer` (one packed
    sample (D,) and the gate's decision), `sample_vectors` (n packed samples, (n, D)) and `num_values`.
    """

    def __init__(self, parts, gate=None):
        self.shapes = [tuple(shape) for shape, _ in parts]
        self.dtypes = [dtype for _, dtype in parts]
        self.sizes = [math.prod(shape) for shape in self.shapes]
        self.D = sum(self.sizes)
        self.gate = gate if gate is not None else NoveltyGate()
        self.offered = 0
        self.written = 0
        self.recognised = 0

    def pack(self, parts):
        """Parts, each (n, *shape), to vectors (n, D)."""
        return torch.cat([p.reshape(len(p), -1).float() for p in parts], dim=1)

    def unpack(self, z):
        """Vectors (n, D) to parts, each (n, *shape) in its original dtype."""
        out = []
        for chunk, shape, dtype in zip(z.split(self.sizes, dim=1), self.shapes, self.dtypes):
            if not dtype.is_floating_point:
                chunk = chunk.round()
            out.append(chunk.reshape((len(z),) + shape).to(dtype))
        return out

    def write(self, parts, novelty):
        """Offer n samples: parts, each (n, *shape); novelty (n,) in [0, 1]."""
        z = self.pack(parts)
        keep = self.gate(novelty)
        for i in range(len(z)):
            self.offered += 1
            self._offer(z[i], bool(keep[i]))

    def sample(self, n):
        """n stored samples drawn for replay, as parts, each (n, *shape)."""
        return self.unpack(self.sample_vectors(n))

    def _offer(self, z, keep):
        raise NotImplementedError

    def sample_vectors(self, n):
        raise NotImplementedError

    def num_values(self):
        raise NotImplementedError


class RingBuffer(Buffer):
    """
    FIFO of `capacity` samples: the replay-only LTM's buffer, with a gate. Sampling is uniform, without replacement
    when the buffer holds enough.
    """

    def __init__(self, capacity, parts, gate=None, seed=0):
        super().__init__(parts, gate)
        self.capacity = capacity
        self.buf = torch.zeros(capacity, self.D)
        self.size = 0
        self.head = 0
        self.gen = torch.Generator().manual_seed(seed)

    def _offer(self, z, keep):
        if not keep:
            return
        self.buf[self.head] = z
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.written += 1

    def sample_vectors(self, n):
        if self.size >= n:
            idx = torch.randperm(self.size, generator=self.gen)[:n]
        else:
            idx = torch.randint(self.size, (n,), generator=self.gen)
        return self.buf[idx]

    def num_values(self):
        return self.buf.numel()


class NeuralBuffer(Buffer):
    """
    Dentate-gyrus-style buffer of `cells` cells, one sample per cell (design: single_stream_paper
    Notes/design/dg_replay_buffer.md). Each cell stores a whole sample vector (D values).

    Similarity: each cell sees `fan_in` fixed random positions of the key (sparse connectivity); its response to a
    sample is the cosine between the sample and the cell's stored vector on those positions. The key is the parts
    listed in `key_parts` (default: all), e.g. the image but not the label. Only cells that have been written
    respond. (A separate projection weight per cell would only duplicate the stored vector on those positions.)

    Utilisation: a trace per cell, decaying by `util_decay` per offered sample, set to 1 when the cell is written
    and credited by 1 when it recognises a sample.

    Offer: if `match_threshold` is set and the best-responding cell responds at least match_threshold, the sample
    counts as already stored: that cell is credited and the sample is not stored again (whatever the gate decided).
    Otherwise, if the gate keeps it, it is written into the least-utilised cell (empty cells first; ties broken by
    least recent activity), overwriting that cell's previous sample. Without recognition the least-utilised cell is
    the least recently written, so the buffer is exactly a FIFO of `cells` samples.

    Sampling: uniform over the cells that have been written, without replacement when there are enough.
    """

    def __init__(self, cells, parts, fan_in=64, key_parts=None, util_decay=0.999, match_threshold=None, gate=None,
                 seed=0):
        super().__init__(parts, gate)
        key_parts = range(len(self.sizes)) if key_parts is None else key_parts
        offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        key_pos = torch.cat([torch.arange(offsets[i], offsets[i] + self.sizes[i]) for i in key_parts])
        if not 1 <= fan_in <= len(key_pos):
            raise ValueError(f"need 1 <= fan_in <= {len(key_pos)} key values")
        self.cells = cells
        self.fan_in = fan_in
        self.util_decay = util_decay
        self.match_threshold = match_threshold
        self.gen = torch.Generator().manual_seed(seed)
        # fixed sparse connectivity: each cell's key positions (seed-generated, not counted as values)
        self.idx = torch.stack([key_pos[torch.randperm(len(key_pos), generator=self.gen)[:fan_in]]
                                for _ in range(cells)])
        self.C = torch.zeros(cells, self.D)
        self.util = torch.zeros(cells, dtype=torch.float64)
        self.stamp = torch.full((cells,), -1, dtype=torch.long)  # offered-sample count at last write or credit
        self.occupied = torch.zeros(cells, dtype=torch.bool)

    def responses(self, z):
        """Cosine response of every cell to packed samples z ((D,) or (n, D)): (cells,) or (n, cells). Cells never
        written respond -inf."""
        zs = z[..., self.idx]
        cs = self.C.gather(1, self.idx)
        r = (zs * cs).sum(-1) / (zs.norm(dim=-1) * cs.norm(dim=-1) + 1e-8)
        return r.masked_fill(~self.occupied, float("-inf"))

    def _offer(self, z, keep):
        self.util.mul_(self.util_decay)
        if self.match_threshold is not None and self.occupied.any():
            r = self.responses(z)
            best = int(r.argmax())
            if r[best] >= self.match_threshold:
                self.util[best] += 1
                self.stamp[best] = self.offered
                self.recognised += 1
                return
        if not keep:
            return
        order = torch.argsort(self.stamp, stable=True)
        cell = int(order[torch.argmin(self.util[order])])  # argmin returns the first minimum: least recent
        self.C[cell] = z
        self.util[cell] = 1.0
        self.stamp[cell] = self.offered
        self.occupied[cell] = True
        self.written += 1

    def sample_vectors(self, n):
        occ = self.occupied.nonzero().flatten()
        if len(occ) >= n:
            pick = occ[torch.randperm(len(occ), generator=self.gen)[:n]]
        else:
            pick = occ[torch.randint(len(occ), (n,), generator=self.gen)]
        return self.C[pick]

    def num_values(self):
        return self.C.numel() + self.util.numel() + self.stamp.numel()
