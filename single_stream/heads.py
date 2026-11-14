"""
Comparison heads on the LTM's features (Dave 10-13 Nov 2026; plan.md, comparison models): fast learners that replace
the LTM's output layer for every awake sample, while the LTM itself (BufferedReplayLTM) learns only in its sleeps, as in
our CLS model. Ported from cifar_main_head_baselines.py (branch main) to the transformer LTM's d_model features and
the stream's classes:

    knn   the samples since the last reset (features, labels), cosine similarity, k nearest vote.
    fly   FlyModel (Shen, Dasgupta & Navlakha 2021): random binary projection to Kenyon cells, top-k_frac WTA,
          Hebbian class weights clipped at 1; no gradients.
    sdmlp SDMLP (Bricken et al. 2023): L2-normalised positive addresses, Top-K with the GABA switch, SGD + CE.

HeadModel gives the heads the same data access as our STM: each live sample, plus `replay` samples per exposure from
the LTM's buffer (without replay a head trained on one cluster's class labels everything as that class). The head
restarts from its initial state at each sleep (`reset_on_sleep`: the LTM's features change in sleep), and comes online
after `delay` exposures, as the STM (--stm-delay). While the head holds nothing since its last reset, the model answers
with the LTM (an empty head has no answer); the model never uses phase or cluster information.
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn

from single_stream.models import StreamModel


class Head:
    name = "head"

    def fit(self, f, y):
        """One online update on features f (n, d) with labels y."""
        raise NotImplementedError

    @torch.no_grad()
    def logits(self, f):
        raise NotImplementedError

    @property
    def empty(self):
        return False

    def num_values(self):
        return 0


class KNNHead(Head):
    """k nearest neighbours (cosine) over the stored samples; scores = votes per class (ties: the nearest wins)."""
    name = "knn"

    def __init__(self, dim, num_classes, k=1, capacity=4096):
        self.k, self.num_classes, self.capacity = k, num_classes, capacity
        self.keys = torch.zeros(0, dim)
        self.labels = torch.zeros(0, dtype=torch.long)

    @property
    def empty(self):
        return len(self.labels) == 0

    @torch.no_grad()
    def fit(self, f, y):
        self.keys = torch.cat([self.keys, F.normalize(f, dim=1)])[-self.capacity:]
        self.labels = torch.cat([self.labels, y])[-self.capacity:]

    @torch.no_grad()
    def logits(self, f):
        sims = F.normalize(f, dim=1) @ self.keys.T
        k = min(self.k, len(self.labels))
        top, idx = sims.topk(k, dim=1)
        out = torch.zeros(len(f), self.num_classes)
        out.scatter_add_(1, self.labels[idx], torch.ones_like(top) + 1e-3 * top)  # votes, nearest breaks ties
        return out

    def num_values(self):
        return self.keys.numel() + self.labels.numel()


class FlyHead(Head):
    """
    FlyModel, as cifar_main_head_baselines.FlyModelHead: f -> min-max scale per row -> KC = R f (R binary, n_response
    ones per row) -> ReLU -> keep the top k_frac -> divide by the row max; W[c] += lr * KC for class c, W <= 1;
    logits = W KC.
    """
    name = "fly"

    def __init__(self, dim, num_classes, n_kc=2000, n_response=16, k_frac=0.05, lr=0.2, seed=0):
        self.k_frac, self.lr = k_frac, lr
        g = torch.Generator().manual_seed(seed)
        cols = torch.rand(n_kc, dim, generator=g).argsort(dim=1)[:, :n_response]
        self.R = torch.zeros(n_kc, dim).scatter_(1, cols, 1.0)
        self.W = torch.zeros(num_classes, n_kc)
        self.seen = 0

    @property
    def empty(self):
        return self.seen == 0

    @torch.no_grad()
    def kc(self, f):
        mn, mx = f.min(dim=1, keepdim=True).values, f.max(dim=1, keepdim=True).values
        f = (f - mn) / torch.where(mx - mn > 0, mx - mn, torch.ones_like(mx))
        kc = torch.clamp(f @ self.R.T, min=0)
        thr = torch.quantile(kc, 1.0 - self.k_frac, dim=1, keepdim=True)
        kc = torch.where(kc < thr, torch.zeros_like(kc), kc)
        mx = kc.max(dim=1, keepdim=True).values
        return kc / torch.where(mx > 0, mx, torch.ones_like(mx))

    @torch.no_grad()
    def fit(self, f, y):
        kc = self.kc(f)
        for c in torch.unique(y):
            self.W[c] += self.lr * kc[y == c].sum(0)
        self.W.clamp_(max=1.0)
        self.seen += len(y)

    @torch.no_grad()
    def logits(self, f):
        return self.kc(f) @ self.W.T

    def num_values(self):
        return self.W.numel()  # R is fixed and seed-generated


class SDMLPHead(Head):
    """
    SDMLP, as cifar_main_head_baselines.SDMLPHead (Bricken's CIFAR settings: no biases, all weights >= 0, fc1 rows
    L2-normalised, Top-K with the GABA switch, gradient clip 1, plain SGD): f -> ReLU -> L2-normalise -> fc1 -> Top-K
    -> purkinje -> logits. GABA counters count activations in training only.
    """
    name = "sdmlp"

    def __init__(self, dim, num_classes, nneurons=1000, k=32, gaba_switch=100, lr=0.05, grad_clip=1.0, seed=0):
        g = torch.random.fork_rng(devices=[])
        with g:
            torch.manual_seed(seed)
            self.fc1 = nn.Linear(dim, nneurons, bias=False)
            self.purkinje = nn.Linear(nneurons, num_classes, bias=False)
        self.k, self.gaba_switch, self.grad_clip = k, max(1, int(gaba_switch)), grad_clip
        self.counters = torch.zeros(1, nneurons)
        self.params = list(self.fc1.parameters()) + list(self.purkinje.parameters())
        self.opt = torch.optim.SGD(self.params, lr=lr)
        self.seen = 0
        self._constrain()

    @property
    def empty(self):
        return self.seen == 0

    @torch.no_grad()
    def _constrain(self):
        for p in self.params:
            p.clamp_(0)
        self.fc1.weight /= torch.norm(self.fc1.weight, dim=1, keepdim=True).clamp_min(1e-12)

    def _forward(self, f, training):
        f = F.relu(f)
        f = f / torch.norm(f, dim=1, keepdim=True).clamp_min(1e-12)
        a = F.relu(self.fc1(f))
        vals, _ = torch.topk(a, min(self.k + 1, a.shape[1]), dim=1, sorted=False)
        gaba = torch.clamp(-1.0 + 2.0 / self.gaba_switch * self.counters, -1.0, 1.0)
        a = F.relu(a - gaba * vals.detach().min(dim=1, keepdim=True).values)
        if training:
            self.counters += (a > 0).sum(dim=0, keepdim=True).detach()
        return self.purkinje(a)

    def fit(self, f, y):
        self.opt.zero_grad()
        F.cross_entropy(self._forward(f, True), y).backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.opt.step()
        self._constrain()
        self.seen += len(y)

    @torch.no_grad()
    def logits(self, f):
        return self._forward(f, False)

    def num_values(self):
        return sum(p.numel() for p in self.params) + self.counters.numel()


def make_head(args, dim, num_classes):
    """The head named by args.head with the --knn-*, --fly-*, --sdmlp-* options (run_stream and stm_frozen)."""
    if args.head == "knn":
        return KNNHead(dim, num_classes, k=args.knn_k, capacity=args.knn_capacity)
    if args.head == "fly":
        return FlyHead(dim, num_classes, n_kc=args.fly_kc, n_response=args.fly_response, k_frac=args.fly_k_frac,
                       lr=args.fly_lr, seed=args.seed)
    if args.head == "sdmlp":
        return SDMLPHead(dim, num_classes, nneurons=args.sdmlp_neurons, k=args.sdmlp_k, gaba_switch=args.sdmlp_gaba,
                         lr=args.sdmlp_lr, seed=args.seed)
    raise ValueError(f"unknown head {args.head}")


class HeadModel(StreamModel):
    """
    The LTM (BufferedReplayLTM) with a fast head replacing its output layer while awake. Per exposure, once online:
    predict with the head (or the LTM if the head is empty), fit the head on the live sample and on `replay` samples
    drawn from the LTM's buffer (fractional rates accumulate), then the LTM's own step; after an LTM update (a sleep)
    the head restarts from its initial state if `reset_on_sleep`. Features are the LTM's (net.run(x)["features"]),
    computed afresh each time.
    """

    def __init__(self, ltm, head, replay=1.0, reset_on_sleep=True, delay=0):
        self.ltm, self.head, self.replay = ltm, head, replay
        self.initial = copy.deepcopy(head)
        self.reset_on_sleep, self.delay = reset_on_sleep, delay
        self.exposures = 0
        self.replay_pending = 0.0
        self.resets = 0

    @property
    def buffer(self):
        return self.ltm.buffer

    @property
    def updates(self):
        return self.ltm.updates

    @property
    def online(self):
        return self.exposures >= self.delay

    @torch.no_grad()
    def _features(self, x):
        self.ltm.net.eval()
        return self.ltm.net.run(x)["features"]

    @torch.no_grad()
    def predict(self, x):
        if not self.online or self.head.empty:
            return self.ltm.predict(x)
        return self.head.logits(self._features(x))

    def observe(self, x, y):
        if not self.online:
            self.exposures += len(x)
            return self.ltm.observe(x, y)
        self.exposures += len(x)
        pred = self.predict(x).argmax(1)
        self.head.fit(self._features(x), y)
        self.replay_pending += self.replay
        if self.ltm.buffer.written == 0:
            self.replay_pending = 0.0
        while self.replay_pending >= 1:
            self.replay_pending -= 1
            (bx, by), _ = self.ltm.buffer.sample(1)
            self.head.fit(self._features(bx), by)
        before = self.ltm.updates
        self.ltm.observe(x, y)
        if self.reset_on_sleep and self.ltm.updates > before:
            self.head = copy.deepcopy(self.initial)
            self.resets += 1
        return pred

    def memory_values(self):
        return self.ltm.memory_values() + self.head.num_values()
