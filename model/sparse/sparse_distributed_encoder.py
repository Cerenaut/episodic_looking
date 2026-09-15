
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

    TODO: Increase capacity by recruiting idle "cells" based on duty cycle tracking.
    - Very slowly, add weight to never-used cells.
    - Possibly remove weight from "synapses" which are never used when cells are active..
    """

    def __init__(
        self,
        input_key_size: int,
        memory_size: int = 65536,
        ensemble_size: int = 1,
        sparsity: int = 32,
        device = None,
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
        scores = (
            input @ self.input_projection.T
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
