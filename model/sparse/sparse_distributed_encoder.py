
import torch
from torch import nn

from util.device import get_device
from util.sparse import (
    get_ensemble_sparse_signed_pairs_projection,
)


class SparseDistributedEncoder(nn.Module):
    """
    Maps inputs to cliques of active cells which should represent or model the input.
    Compatible with minibatch learning but does not require it.
    The representation is sparse and distributed for combinatoric capacity and robust
    value retention. Ensemble projection is used to increase orthogonality of similar
    input and thereby improve capacity.

    Recruitment of idle cells (optional, off when recruit_rate = 0; Dave, 1 Nov 2026). The top-k selection scores
    use the fixed projection plus a recruitment weight, which starts at zero. On each `recruit` call (training only,
    by the owner, with the key it just encoded): cells in the active clique are stamped active; every cell idle for
    more than recruit_idle_steps calls (with probability recruit_prob each) has recruit_rate x (random uniform
    amounts in [0, 1) x the normalised current key) added to its recruitment weight, which raises its score for
    this key until it wins a place in a clique. A cell that has been active gets nothing until it has been idle for
    recruit_idle_steps again. Only the clique selection changes; the learned parameters of the models that use the
    mask are untouched. The recruitment weights and activity stamps depend on the data, so they count as values.
    """

    def __init__(
        self,
        input_key_size: int,
        memory_size: int = 65536,
        ensemble_size: int = 1,
        sparsity: int = 32,
        device = None,
        recruit_rate: float = 0.0,
        recruit_idle_steps: int = 10000,
        recruit_prob: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()

        self.device = device
        if self.device is None:
            self.device = get_device()

        self.input_key_size = input_key_size
        self.memory_size = memory_size
        self.ensemble_size = ensemble_size
        self.sparsity = sparsity

        # Create fixed projection of input
        E = self.ensemble_size
        Z = self.memory_size
        I = self.input_key_size
        projection = self._create_input_projection()  # [E, Z, I]
        input_projection_2d = (
            projection
            .reshape(E * Z, I)
        )
        self.register_buffer("input_projection", input_projection_2d)

        self.recruit_rate = recruit_rate
        self.recruit_idle_steps = recruit_idle_steps
        self.recruit_prob = recruit_prob
        # Usage counts are diagnostics only (cell utilisation), kept whether or not recruitment is on
        self.register_buffer("active_count", torch.zeros(Z, dtype=torch.long, device=self.device))
        if recruit_rate > 0:
            self.register_buffer("recruit_weight", torch.zeros(E * Z, I, device=self.device))
            self.register_buffer("last_active", torch.zeros(Z, dtype=torch.long, device=self.device))
            self.register_buffer("recruit_calls", torch.zeros((), dtype=torch.long, device=self.device))
            self.recruit_gen = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self.recruit_weight = None

    def _create_input_projection(self):
        # Fixed random projection (not trained) row-normalised random hyperplane hash
        projection = get_ensemble_sparse_signed_pairs_projection(
            num_ensembles = self.ensemble_size,
            input_size = self.input_key_size,
            hidden_size = self.memory_size,
            requires_grad = False,
            device = self.device,
        )
        return projection

    def encode(self, input: torch.Tensor):
        return self.encode_k(input, k=self.sparsity)

    def encode_k(self, input: torch.Tensor, k:int):
        """
        Single-stage encode with true ensemble competition.

        input: [B, input_size]
        returns indices: [B, sparsity], mask [B, H]
        """
        B, I = input.shape
        E = self.ensemble_size
        Z = self.memory_size
        
        # Vectorised projection for all ensembles
        projection = self.input_projection
        if self.recruit_weight is not None:
            projection = projection + self.recruit_weight
        scores = (
            input @ projection.T
        ).view(B, E, Z)  # faster because uses standard GEMM format

        # Scores shape: [B, E, Z]
        
        # Top-k per ensemble, only these considered as potential winners
        _, ensemble_topk_indices = torch.topk(scores, k, dim=2) # [B, E, k]

        # Build union mask of winners
        mask = torch.zeros(B, Z, dtype=torch.bool, device=self.device)
        idx_flat = ensemble_topk_indices.reshape(B, -1)  # flatten [B, E*k]
        mask.scatter_(1, idx_flat, True)

        # Re-score only selected cells (using full ensemble info)
        # This is important: don't just take union
        mask_value = float('-inf')
        scores_masked = scores.masked_fill(~mask.unsqueeze(1), mask_value)

        # Aggregate AFTER masking (preserves ensemble contribution)
        scores_combined = scores_masked.max(dim=1).values  # [B, Z], max over E

        # Final top-k
        _, final_indices = torch.topk(scores_combined, k, dim=1)
        return final_indices

    @torch.no_grad()
    def record_active(self, active_indices: torch.Tensor):
        """Count clique membership (diagnostics). active_indices: [B, sparsity]."""
        self.active_count.index_add_(0, active_indices.flatten(),
                                     torch.ones(active_indices.numel(), dtype=torch.long, device=self.device))

    @torch.no_grad()
    def recruit(self, input: torch.Tensor, active_indices: torch.Tensor) -> int:
        """
        One recruitment update (training only). input: [B, input_size] keys just encoded; active_indices:
        [B, sparsity] their cliques. Returns the number of cells that received recruitment weight.
        """
        if self.recruit_weight is None:
            return 0
        self.recruit_calls += 1
        t = int(self.recruit_calls)
        self.last_active[active_indices.flatten()] = t
        idle = (t - self.last_active) > self.recruit_idle_steps
        if self.recruit_prob < 1:
            idle &= torch.rand(self.memory_size, generator=self.recruit_gen, device=self.device) < self.recruit_prob
        cells = idle.nonzero().flatten()
        if len(cells) == 0:
            return 0
        key = input.mean(0)
        key = key / (key.norm() + 1e-8)
        E, Z = self.ensemble_size, self.memory_size
        rows = (torch.arange(E, device=self.device)[:, None] * Z + cells[None, :]).flatten()
        amounts = torch.rand(len(rows), self.input_key_size, generator=self.recruit_gen, device=self.device)
        self.recruit_weight[rows] += self.recruit_rate * amounts * key
        return len(cells)

    def utilisation(self) -> dict:
        """Diagnostics: fraction of cells ever in a clique, and the normalised entropy of clique membership."""
        c = self.active_count.double()
        total = c.sum()
        if total == 0:
            return {"cells_used": 0.0, "usage_entropy": 0.0}
        p = c[c > 0] / total
        return {"cells_used": float((c > 0).double().mean()),
                "usage_entropy": float(-(p * p.log()).sum() / torch.log(torch.tensor(float(self.memory_size))))}

    def num_values(self) -> int:
        """Data-dependent values: recruitment weights and activity stamps (the projection is fixed, seed-made)."""
        if self.recruit_weight is None:
            return 0
        return self.recruit_weight.numel() + self.last_active.numel()
