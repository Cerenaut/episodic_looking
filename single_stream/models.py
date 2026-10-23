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
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x)


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

    def __init__(self, net, lr, steps=1, momentum=0.0, weight_decay=0.0):
        self.net = net
        self.steps = steps
        self.opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def _train_batch(self, x, y):
        pred = None
        self.net.train()
        for _ in range(self.steps):
            logits = self.net(x)
            if pred is None:
                pred = logits.detach().argmax(1)
            loss = F.cross_entropy(logits, y)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
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
                 seed=0):
        super().__init__(net, lr, steps, momentum, weight_decay)
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
            loss = F.cross_entropy(logits, by)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
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
