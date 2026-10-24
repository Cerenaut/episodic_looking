"""
Replay buffers for replay-trained models. Every exposure is offered to the buffer with a novelty score in [0, 1]
from the model (in task 1, the LTM's error on the live sample, 1 - p(y | x), before any update); the buffer decides
whether to store it. The model draws samples from the buffer for replay. Buffers never receive group, cluster or
task-switch information.

Memory is counted in values (plan.md section 4): every value whose content depends on the data seen. Fixed,
seed-generated weights are not counted.
"""
import torch
import torch.nn.functional as F


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


class Buffer:

    def __init__(self, gate=None):
        self.gate = gate if gate is not None else NoveltyGate()
        self.offered = 0
        self.written = 0
        self.recognised = 0

    def write(self, x, y, novelty):
        """Offer a batch of exposures (x: (n, D) in [0, 1], y: (n,), novelty: (n,) in [0, 1])."""
        keep = self.gate(novelty)
        for i in range(len(x)):
            self.offered += 1
            self._offer(x[i], y[i], bool(keep[i]))

    def _offer(self, x, y, keep):
        if keep:
            self._store(x, y)

    def _store(self, x, y):
        raise NotImplementedError

    def sample(self, n):
        """n samples for replay: inputs (n, D) in [0, 1] and labels (n,)."""
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def nbytes(self):
        raise NotImplementedError


class RingBuffer(Buffer):
    """
    FIFO of `capacity` slots: the replay-only LTM's buffer, with a gate. Inputs in [0, 1] are stored as 8-bit (exact
    for 8-bit images). Sampling is uniform, without replacement when the buffer holds enough.
    """

    def __init__(self, capacity, input_size, gate=None, store_uint8=True, seed=0):
        super().__init__(gate)
        self.capacity = capacity
        self.input_size = input_size
        self.store_uint8 = store_uint8
        self.buf_x = torch.zeros(capacity, input_size, dtype=torch.uint8 if store_uint8 else torch.float32)
        self.buf_y = torch.zeros(capacity, dtype=torch.long)
        self.size = 0
        self.head = 0
        self.gen = torch.Generator().manual_seed(seed)

    def _store(self, x, y):
        self.buf_x[self.head] = torch.round(x * 255.0).clamp(0, 255).to(torch.uint8) if self.store_uint8 else x
        self.buf_y[self.head] = y
        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.written += 1

    def sample(self, n):
        if self.size >= n:
            idx = torch.randperm(self.size, generator=self.gen)[:n]
        else:
            idx = torch.randint(self.size, (n,), generator=self.gen)
        x = self.buf_x[idx]
        return (x.float() / 255.0 if self.store_uint8 else x), self.buf_y[idx]

    def values(self):
        return self.capacity * (self.input_size + 1)

    def nbytes(self):
        return self.buf_x.numel() * self.buf_x.element_size() + self.buf_y.numel() * self.buf_y.element_size()


class SeparatedBuffer(Buffer):
    """
    Dentate-gyrus-style buffer of `cells` cells (design: single_stream_paper Notes/design/dg_replay_buffer.md).

    Each cell sees `fan_in` fixed random input pixels (sparse connectivity); its response to an input is the cosine
    between its projection weights P and the input on those pixels. A memory is the full sample and label,
    z = [x, label_scale * onehot(y)], held in the content weights C of the k cells that code it (their C rows sum
    to z). Each cell keeps a utilisation trace that decays by `util_decay` per offered exposure.

    allocation "least_used": a stored sample recruits the k least-utilised cells (empty cells first). Their
    projection weights are set to the input on their pixels (one-shot) and their content is overwritten; their
    utilisation is set to 1. Projection weights change only for recruited cells, never for ordinary inputs.
    Without recognition (match_threshold None) the least-utilised cell is the least recently written, so k = 1 is
    exactly a FIFO of `cells` samples, and k > 1 is a FIFO of cells // k samples each held k times.

    allocation "response": the classical DG. P is fixed random (seed-generated, never learned). A stored sample is
    coded by the k cells with the highest response minus `homeostasis` x utilisation (usage balancing) and written
    into their content by the normalised delta rule, C[cells] += (z - sum C[cells]) / k, so after the write the
    code recalls z exactly. Codes of different samples may share cells: combinatorial, with interference.

    Recognition (match_threshold, either allocation): an offered input whose top k occupied cells all respond at
    least match_threshold counts as already stored: those cells' utilisation is credited by 1 and the input is not
    stored again, whether or not the gate would have kept it.

    Sampling: a random occupied cell, read out (its content x k), then `settle` times: re-code the recalled x by
    response (top k occupied cells, no homeostasis) and read the sum of their content. k = 1 reads the cell directly.
    Default settle 0: a least_used cell holds z / k of one memory, so the read-out is already exact, and with the
    response allocation the write-time code includes the usage penalty, so re-coding by response alone picks other
    cells and makes recall worse (test_response_allocation_recalls_exactly_without_settling). Settling only helps
    when cells are shared and codes are recoverable from the input (homeostasis 0).
    """

    def __init__(self, cells, input_size, num_classes, k=1, fan_in=64, allocation="least_used", label_scale=1.0,
                 util_decay=0.999, homeostasis=1.0, match_threshold=None, settle=0, gate=None, seed=0):
        super().__init__(gate)
        if allocation not in ("least_used", "response"):
            raise ValueError(f"unknown allocation {allocation}")
        if not 1 <= k <= cells or not 1 <= fan_in <= input_size:
            raise ValueError("need 1 <= k <= cells and 1 <= fan_in <= input_size")
        self.cells = cells
        self.input_size = input_size
        self.num_classes = num_classes
        self.k = k
        self.fan_in = fan_in
        self.allocation = allocation
        self.label_scale = label_scale
        self.util_decay = util_decay
        self.homeostasis = homeostasis
        self.match_threshold = match_threshold
        self.settle = settle
        self.gen = torch.Generator().manual_seed(seed)
        # fixed sparse connectivity: each cell's input pixels
        self.idx = torch.stack([torch.randperm(input_size, generator=self.gen)[:fan_in] for _ in range(cells)])
        if allocation == "response":
            self.P = torch.randn(cells, fan_in, generator=self.gen)
        else:
            self.P = torch.zeros(cells, fan_in)
        self.C = torch.zeros(cells, input_size + num_classes)
        self.util = torch.zeros(cells, dtype=torch.float64)
        self.stamp = torch.full((cells,), -1, dtype=torch.long)  # offered-exposure count at last write/credit
        self.occupied = torch.zeros(cells, dtype=torch.bool)

    def responses(self, x):
        """Cosine response of every cell to x ((D,) or (n, D)) on its own pixels: (N,) or (n, N)."""
        xs = x[..., self.idx]
        num = (xs * self.P).sum(-1)
        return num / (self.P.norm(dim=-1) * xs.norm(dim=-1) + 1e-8)

    def _top_occupied(self, r):
        return r.masked_fill(~self.occupied, float("-inf")).topk(self.k, dim=-1)

    def _offer(self, x, y, keep):
        self.util.mul_(self.util_decay)
        r = None
        if self.match_threshold is not None and int(self.occupied.sum()) >= self.k:
            r = self.responses(x)
            top = self._top_occupied(r)
            if top.values.min() >= self.match_threshold:
                self.util[top.indices] += 1
                self.stamp[top.indices] = self.offered
                self.recognised += 1
                return
        if keep:
            self._store(x, y, r)

    def _store(self, x, y, r=None):
        z = torch.cat([x, self.label_scale * F.one_hot(y, self.num_classes).float()])
        if self.allocation == "least_used":
            # least utilised first; ties (empty cells, or traces decayed to 0) broken by least recent activity
            order = torch.argsort(self.stamp, stable=True)
            cells = order[torch.argsort(self.util[order], stable=True)][:self.k]
            self.P[cells] = x[self.idx[cells]]
            self.C[cells] = z / self.k
            self.util[cells] = 1.0
        else:
            r = self.responses(x) if r is None else r
            cells = (r - self.homeostasis * self.util.float()).topk(self.k).indices
            self.C[cells] += (z - self.C[cells].sum(0)) / self.k
            self.util[cells] += 1.0
        self.occupied[cells] = True
        self.stamp[cells] = self.offered
        self.written += 1

    def recall(self, start):
        """Recall from start cells (n,): returns z (n, D + L)."""
        z = self.C[start] * self.k
        if self.k > 1:
            for _ in range(self.settle):
                cells = self._top_occupied(self.responses(z[:, :self.input_size].clamp(0, 1))).indices
                z = self.C[cells].sum(1)
        return z

    def sample(self, n):
        occ = self.occupied.nonzero().flatten()
        if len(occ) >= n:
            start = occ[torch.randperm(len(occ), generator=self.gen)[:n]]
        else:
            start = occ[torch.randint(len(occ), (n,), generator=self.gen)]
        z = self.recall(start)
        return z[:, :self.input_size].clamp(0, 1), z[:, self.input_size:].argmax(1)

    def values(self):
        # P counts only once it is learned (least_used); the response allocation's P is fixed random
        learned_p = self.P.numel() if self.allocation == "least_used" else 0
        return learned_p + self.C.numel() + self.util.numel() + self.stamp.numel()

    def nbytes(self):
        learned_p = self.P.numel() * 4 if self.allocation == "least_used" else 0
        return learned_p + self.C.numel() * 4 + (self.util.numel() + self.stamp.numel()) * 8
