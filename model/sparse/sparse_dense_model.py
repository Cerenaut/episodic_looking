from dataclasses import dataclass

import torch

from model.dense import DenseModel, DenseModelConfig


@dataclass
class SparseActivationDenseModelConfig(DenseModelConfig):
    hidden_size_factor: int = 2  # e.g. 2 or 4
    weight_decay_factor: float = 0.0  # Suggested start 1e-4, then explore 1e-5 .. 5e-4

class SparseActivationDenseModel(DenseModel):
    """
    A DenseModel with sparse activation in the hidden layers.
    """

    def __init__(self, config: SparseActivationDenseModelConfig):
        super().__init__(config)

    def get_active_mask_hidden_size(self):
        mask_hidden_size = self.config.hidden_size // self.config.hidden_size_factor
        return mask_hidden_size
    
    def create_active_mask(self, x, active_indices):
        """
        Returns mask of shape [B, mask_hidden_size]
        mask_hidden_size = self.get_mask_hidden_size()
        """
        B, k = active_indices.shape
        M = self.get_active_mask_hidden_size()
        F = self.config.hidden_size_factor
        mask = torch.zeros(
            B,
            M,
            device=x.device,
            dtype=x.dtype,
            requires_grad=False,
        )
        mask.scatter_(1, active_indices, 1.0)  # scatter 1.0 into dim=1 at indices

        mask = mask.unsqueeze(2).expand(B, M, F)  # [B,M] --> [B,M,F]
        mask = mask.reshape(B, M * F)  # [B,M,F] --> [B,M*F] == [B, H]
        return mask

    def forward(self, x, active_mask: torch.Tensor):

        # Input layer norm - some interference but hopefully minor
        if self.ln is not None:
            x = self.ln(x)

        # Dropout for regularization *within* the active clique and its use of inputs.
        if self.dropout_input is not None:
            x = self.dropout_input(x)

        for layer in range(self.config.layers):

            # Apply matmul for layer
            fc = self.layers[layer]
            x = fc(x)

            is_hidden = layer < (self.config.layers - 1)

            # Activation
            f = self.f_hidden if is_hidden else self.f_output
            if f is not None:
                x = f(x)

            # Apply sparsity only on hidden layers. 
            # Output layer will have zero grad from masked final hidden layer cells
            # This means non-interference between masked cliques
            if is_hidden:
                x = x * active_mask

            # Dropout
            if is_hidden and self.config.hidden_dropout > 0:
                d = self.dropout_layers[layer]
                x = d(x)

        return x

    def active_weight_decay(self, active_mask: torch.Tensor, decay_factor:float|None=None):
        """
        Applies weight decay only to cells corresponding to active neurons.
        
        active_mask: [B, hidden_size]
            1 for active neurons, 0 otherwise

        decay: float
            weight decay coefficient
        """
        if decay_factor is None:
            decay_factor = self.config.weight_decay_factor
        if decay_factor <= 0:
            return
        
        for layer in range(self.config.layers):

            # Apply matmul for layer
            fc = self.layers[layer]

            # Apply decay only to active columns
            with torch.no_grad():
                # [in_features]
                # Decay if cell was used in any minibatch sample
                # broadcast to [out_features, in_features]
                if layer < (self.config.layers -1):
                    n = 1
                else:  # output layer
                    n = 0
                col_mask = active_mask.any(dim=0).float().unsqueeze(n)
                fc.weight *= (1 - decay_factor * col_mask)