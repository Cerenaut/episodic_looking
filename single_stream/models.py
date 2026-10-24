"""
Stream models. The task harness treats every model as a black box: it calls observe(x, y) once per stream step
and predict(x) for evaluation. Gradient steps per exposure, replay and any other memory mechanism are internal.
Models never receive group, cluster or task-switch information.
"""
import torch
from torch import nn
import torch.nn.functional as F


class StreamModel:

    def observe(self, x, y):
        """Learn from one stream step. Returns the predicted labels made before any update (online accuracy)."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, x):
        """Logits for a batch, with no learning."""
        raise NotImplementedError

    def memory_bytes(self):
        """Bytes stored by the model: parameters plus any stored samples (optimizer state not counted)."""
        raise NotImplementedError


def param_bytes(net):
    return sum(p.numel() * p.element_size() for p in net.parameters())


class MLP(nn.Module):
    """Plain ReLU MLP with Kaiming init (Dohare et al. 2024 online permuted MNIST: 3 hidden layers of 2000)."""

    def __init__(self, input_size, widths, num_classes):
        super().__init__()
        sizes = [input_size] + list(widths)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = nn.Linear(sizes[-1], num_classes)
        for layer in list(self.hidden) + [self.out]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        self.acts = []
        for layer in self.hidden:
            x = F.relu(layer(x))
            self.acts.append(x.detach())
        return self.out(x)


class ContinualBackprop:
    """
    Continual backprop (Dohare et al. 2024, Nature; generate-and-test on the hidden units of an MLP). After each
    optimizer step: update each hidden unit's contribution utility, a running average (decay `decay`) of
    |activation| x sum of |outgoing weights|, and its running mean activation; units older than `maturity` steps
    are eligible; a fractional count accumulates at `rate` x (number eligible) per step, and when it reaches 1 the
    eligible units with the lowest bias-corrected utility are replaced: incoming weights re-initialised (Kaiming
    normal, zero bias), the unit's mean contribution folded into the next layer's bias, outgoing weights zeroed,
    utility and age reset. Phase-blind: runs every step, independent of the stream's structure.
    """

    def __init__(self, net, rate=1e-4, decay=0.99, maturity=100):
        self.net = net
        self.rate = rate
        self.decay = decay
        self.maturity = maturity
        sizes = [layer.out_features for layer in net.hidden]
        self.util = [torch.zeros(n) for n in sizes]
        self.mean_act = [torch.zeros(n) for n in sizes]
        self.age = [torch.zeros(n, dtype=torch.long) for n in sizes]
        self.pending = [0.0 for _ in sizes]
        self.replaced = 0

    def _next_layer(self, i):
        layers = list(self.net.hidden) + [self.net.out]
        return layers[i + 1]

    @torch.no_grad()
    def step(self):
        d = self.decay
        for i, layer in enumerate(self.net.hidden):
            nxt = self._next_layer(i)
            act = self.net.acts[i].mean(0)
            contribution = act.abs() * nxt.weight.abs().sum(0)
            self.util[i].mul_(d).add_((1 - d) * contribution)
            self.mean_act[i].mul_(d).add_((1 - d) * act)
            self.age[i] += 1

            eligible = self.age[i] > self.maturity
            n_eligible = int(eligible.sum())
            if n_eligible == 0:
                continue
            self.pending[i] += self.rate * n_eligible
            if self.pending[i] < 1:
                continue
            k = min(int(self.pending[i]), n_eligible)
            self.pending[i] -= k

            correction = 1 - d ** self.age[i].double()
            corrected = (self.util[i] / correction).float()
            corrected[~eligible] = float("inf")
            units = torch.topk(corrected, k, largest=False).indices

            mean = (self.mean_act[i][units] / correction[units].float())
            nxt.bias += nxt.weight[:, units] @ mean
            nxt.weight[:, units] = 0.0
            new_w = torch.empty(k, layer.in_features)
            nn.init.kaiming_normal_(new_w, nonlinearity="relu")
            layer.weight[units] = new_w
            layer.bias[units] = 0.0
            self.util[i][units] = 0.0
            self.mean_act[i][units] = 0.0
            self.age[i][units] = 0
            self.replaced += k


class ResidualMLP(nn.Module):
    """Pre-LayerNorm residual MLP: a ResNet without the convolutions. LayerNorm, not BatchNorm, because b = 1."""

    def __init__(self, input_size, width, blocks, num_classes, expansion=1):
        super().__init__()
        self.inp = nn.Linear(input_size, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * expansion), nn.ReLU(),
                          nn.Linear(width * expansion, width))
            for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(F.relu(self.norm(x)))


def build_net(name, input_size, num_classes, widths=(2000, 2000, 2000), blocks=3):
    if name == "mlp":
        return MLP(input_size, widths, num_classes)
    if name == "resmlp":
        return ResidualMLP(input_size, widths[0], blocks, num_classes)
    raise ValueError(f"unknown net {name}")


class LTMOnly(StreamModel):
    """The LTM trained directly on the stream with SGD, `steps` gradient steps per exposure."""

    def __init__(self, net, lr, steps=1, momentum=0.0, weight_decay=0.0, cbp=None):
        self.net = net
        self.steps = steps
        self.opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.cbp = cbp  # optional ContinualBackprop, applied after every optimizer step

    def _step(self, loss):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.cbp is not None:
            self.cbp.step()

    def _train_batch(self, x, y):
        pred = None
        self.net.train()
        for _ in range(self.steps):
            logits = self.net(x)
            if pred is None:
                pred = logits.detach().argmax(1)
            self._step(F.cross_entropy(logits, y))
        return pred

    def observe(self, x, y):
        return self._train_batch(x, y)

    @torch.no_grad()
    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def memory_bytes(self):
        return param_bytes(self.net)


class ReplayLTM(LTMOnly):
    """
    LTM + experience replay: a reservoir buffer of `capacity` samples; each gradient step trains on the current
    sample(s) plus `replay_batch` samples drawn uniformly from the buffer (fresh draw per step).
    """

    def __init__(self, net, lr, capacity, replay_batch, input_shape, steps=1, momentum=0.0, weight_decay=0.0,
                 seed=0, cbp=None):
        super().__init__(net, lr, steps, momentum, weight_decay, cbp)
        self.capacity = capacity
        self.replay_batch = replay_batch
        self.buf_x = torch.zeros((capacity,) + tuple(input_shape))
        self.buf_y = torch.zeros(capacity, dtype=torch.long)
        self.size = 0
        self.seen = 0
        self.gen = torch.Generator().manual_seed(seed)

    def observe(self, x, y):
        pred = None
        self.net.train()
        for _ in range(self.steps):
            bx, by = x, y
            if self.size > 0 and self.replay_batch > 0:
                idx = torch.randint(self.size, (self.replay_batch,), generator=self.gen)
                bx = torch.cat([x, self.buf_x[idx]])
                by = torch.cat([y, self.buf_y[idx]])
            logits = self.net(bx)
            if pred is None:
                pred = logits[:len(x)].detach().argmax(1)
            self._step(F.cross_entropy(logits, by))
        self._reservoir_add(x, y)
        return pred

    def _reservoir_add(self, x, y):
        for i in range(len(x)):
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(torch.randint(self.seen + 1, (1,), generator=self.gen))
            if slot < self.capacity:
                self.buf_x[slot] = x[i]
                self.buf_y[slot] = y[i]
            self.seen += 1

    def memory_bytes(self):
        return param_bytes(self.net) + self.buf_x.numel() * self.buf_x.element_size() \
            + self.buf_y.numel() * self.buf_y.element_size()


class ReplayOnlyLTM(StreamModel):
    """
    The LTM trained only from a replay buffer, never directly on the live stream. Each exposure is stored; every
    `period` exposures the model takes `steps` gradient steps, each on `batch` samples drawn uniformly (without
    replacement when the buffer holds enough) from the buffer. Replay ratio = steps x batch / period
    sample-gradients per exposure.

    policy "fifo": a ring buffer of `capacity` slots holding the most recent exposures (age-based forgetting).
    policy "reservoir": reservoir sampling (a uniform sample of everything seen).
    store_uint8: store inputs in [0, 1] as 8-bit (exact for 8-bit images such as MNIST); memory counts the bytes
    actually stored.
    """

    def __init__(self, net, lr, capacity, batch, period, input_shape, steps=1, policy="fifo", store_uint8=True,
                 momentum=0.0, weight_decay=0.0, seed=0, cbp=None):
        if policy not in ("fifo", "reservoir"):
            raise ValueError(f"unknown replay policy {policy}")
        self.net = net
        self.capacity = capacity
        self.batch = batch
        self.period = period
        self.steps = steps
        self.policy = policy
        self.store_uint8 = store_uint8
        self.opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.cbp = cbp
        dtype = torch.uint8 if store_uint8 else torch.float32
        self.buf_x = torch.zeros((capacity,) + tuple(input_shape), dtype=dtype)
        self.buf_y = torch.zeros(capacity, dtype=torch.long)
        self.size = 0
        self.seen = 0
        self.head = 0
        self.pending = 0
        self.updates = 0
        self.gen = torch.Generator().manual_seed(seed)

    def _encode(self, x):
        if not self.store_uint8:
            return x
        q = torch.round(x * 255.0)
        return q.clamp(0, 255).to(torch.uint8)

    def _decode(self, x):
        return x.float() / 255.0 if self.store_uint8 else x

    def _store(self, x, y):
        x = self._encode(x)
        for i in range(len(x)):
            if self.policy == "fifo":
                slot = self.head
                self.head = (self.head + 1) % self.capacity
                self.size = min(self.size + 1, self.capacity)
            elif self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                slot = int(torch.randint(self.seen + 1, (1,), generator=self.gen))
            if slot < self.capacity:
                self.buf_x[slot] = x[i]
                self.buf_y[slot] = y[i]
            self.seen += 1

    def _replay_step(self):
        if self.size >= self.batch:
            idx = torch.randperm(self.size, generator=self.gen)[:self.batch]
        else:
            idx = torch.randint(self.size, (self.batch,), generator=self.gen)
        self.net.train()
        loss = F.cross_entropy(self.net(self._decode(self.buf_x[idx])), self.buf_y[idx])
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.cbp is not None:
            self.cbp.step()
        self.updates += 1

    def observe(self, x, y):
        with torch.no_grad():
            self.net.eval()
            pred = self.net(x).argmax(1)
        self._store(x, y)
        self.pending += len(x)
        while self.pending >= self.period:
            self.pending -= self.period
            for _ in range(self.steps):
                self._replay_step()
        return pred

    @torch.no_grad()
    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def memory_bytes(self):
        return param_bytes(self.net) + self.buf_x.numel() * self.buf_x.element_size() \
            + self.buf_y.numel() * self.buf_y.element_size()


class BufferedReplayLTM(StreamModel):
    """
    The replay-only LTM with a pluggable buffer (single_stream/buffers.py). Each exposure is offered to the buffer
    with a novelty score: the LTM's error on it, 1 - p(y | x), from the same pre-update forward pass that gives the
    online prediction. Every `period` exposures the LTM takes `steps` gradient steps, each on `batch` samples drawn
    from the buffer; it is never trained on the live sample. With RingBuffer and no gate this is ReplayOnlyLTM.
    """

    def __init__(self, net, lr, buffer, batch, period, steps=1, momentum=0.0, weight_decay=0.0, cbp=None):
        self.net = net
        self.buffer = buffer
        self.batch = batch
        self.period = period
        self.steps = steps
        self.opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.cbp = cbp
        self.pending = 0
        self.updates = 0

    @staticmethod
    def novelty(logits, y):
        return 1.0 - F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)

    def _replay_step(self):
        bx, by = self.buffer.sample(self.batch)
        self.net.train()
        loss = F.cross_entropy(self.net(bx), by)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.cbp is not None:
            self.cbp.step()
        self.updates += 1

    def observe(self, x, y):
        with torch.no_grad():
            self.net.eval()
            logits = self.net(x)
        self.buffer.write(x, y, self.novelty(logits, y))
        self.pending += len(x)
        while self.pending >= self.period:
            self.pending -= self.period
            if self.buffer.written == 0:
                continue  # the gate has stored nothing yet
            for _ in range(self.steps):
                self._replay_step()
        return logits.argmax(1)

    @torch.no_grad()
    def predict(self, x):
        self.net.eval()
        return self.net(x)

    def memory_values(self):
        return sum(p.numel() for p in self.net.parameters()) + self.buffer.values()

    def memory_bytes(self):
        return param_bytes(self.net) + self.buffer.nbytes()
