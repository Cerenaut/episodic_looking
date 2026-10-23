import torch
from torch import nn


class InputConditioner(nn.Module):
    """
    Conditions one step's STM observation (a concatenation of parts, e.g. [encoding, bias, logits]) with a fixed,
    sample-independent transform: every image gets the same transform whatever its minibatch, so its address is the
    same in any batch (and at batch 1).

    Methods:
        none          identity
        centre        subtract a per-feature mean
        centre-scale  centre, then divide each part by its expected (centred) norm, so the parts weigh equally
        layernorm     per sample, per part: subtract the part's mean and divide by its std (no statistics)

    Statistics are a running average over the observations seen in training, estimated during a burn-in of
    burnin_samples observations and then frozen (or, if freeze is False, tracked thereafter with an EMA of rate momentum).
    They are buffers, so they are saved with the STM and frozen statistics carry over to later phases.
    """

    METHOD_NONE = "none"
    METHOD_CENTRE = "centre"
    METHOD_CENTRE_SCALE = "centre-scale"
    METHOD_LAYERNORM = "layernorm"
    METHODS = [METHOD_NONE, METHOD_CENTRE, METHOD_CENTRE_SCALE, METHOD_LAYERNORM]

    def __init__(
        self,
        part_sizes: list[int],
        method: str,
        burnin_samples: int,
        freeze: bool = True,
        momentum: float = 0.001,
        eps: float = 1e-5,
        device=None,
    ):
        super().__init__()
        if method not in InputConditioner.METHODS:
            raise ValueError(f"Unknown conditioning method: {method}")
        self.method = method
        self.part_sizes = list(part_sizes)
        self.burnin_samples = burnin_samples
        self.freeze = freeze
        self.momentum = momentum
        self.eps = eps
        size = sum(self.part_sizes)
        P = len(self.part_sizes)
        part_index = torch.cat([torch.full((n,), p, dtype=torch.long) for p, n in enumerate(self.part_sizes)])
        self.register_buffer("part_index", part_index.to(device), persistent=False)
        self.register_buffer("mean", torch.zeros(size, device=device))
        self.register_buffer("part_sq_norm", torch.zeros(P, device=device))  # E ||x_p||^2 (uncentred)
        self.register_buffer("samples", torch.zeros((), dtype=torch.long, device=device))

    def uses_statistics(self) -> bool:
        return self.method in (InputConditioner.METHOD_CENTRE, InputConditioner.METHOD_CENTRE_SCALE)

    def is_frozen(self) -> bool:
        return self.freeze and int(self.samples) >= self.burnin_samples

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """x: [B, size] observations seen in training. Returns True on the update that froze the statistics."""
        if not self.uses_statistics() or x.shape[0] == 0 or self.is_frozen():
            return False
        x = x.detach().float()
        batch_mean = x.mean(0)
        batch_sq = torch.zeros_like(self.part_sq_norm).index_add_(0, self.part_index, (x * x).mean(0))
        n, b = int(self.samples), x.shape[0]
        if n < self.burnin_samples:
            rate = b / (n + b)  # exact running average over the burn-in
        else:
            rate = self.momentum  # only when not freezing
        self.mean.lerp_(batch_mean, rate)
        self.part_sq_norm.lerp_(batch_sq, rate)
        self.samples += b
        return self.is_frozen()

    def part_scale(self) -> torch.Tensor:
        """1 / expected centred norm of each part, per feature; 1 where a part has no variance (e.g. bias at step 1)."""
        centred_sq = self.part_sq_norm - torch.zeros_like(self.part_sq_norm).index_add_(
            0, self.part_index, self.mean * self.mean)
        norm = centred_sq.clamp_min(0).sqrt()
        scale = torch.where(norm > self.eps, 1.0 / norm.clamp_min(self.eps), torch.ones_like(norm))
        return scale[self.part_index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., size]"""
        if self.method == InputConditioner.METHOD_NONE:
            return x
        if self.method == InputConditioner.METHOD_CENTRE:
            return x - self.mean
        if self.method == InputConditioner.METHOD_CENTRE_SCALE:
            return (x - self.mean) * self.part_scale()
        # layernorm, per part
        parts = torch.split(x, self.part_sizes, dim=-1)
        out = []
        for p in parts:
            m = p.mean(-1, keepdim=True)
            sd = p.std(-1, keepdim=True, unbiased=False)
            out.append((p - m) / (sd + self.eps))
        return torch.cat(out, dim=-1)
