from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch


@dataclass
class ObservationHistoryConfig:
    batch_size: int = 0
    history_size: int = 0
    observation_size: int = 0
    reset_value: float = 0

class ObservationHistory:
    """
    Maintains a rolling observation history:
        history shape = [B, H, O]

    We assume no recurrent grads ie clone() has no grads from previous steps.

    - new obs have shape [B, O]
    - reset_mask has shape [B] type bool
    """

    def __init__(self, config, device=None, dtype=torch.float32):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.reset()

    @torch.no_grad()
    def clone(self) -> ObservationHistory:
        temp = self.history
        self.history = None
        history_copy = deepcopy(self)
        self.history = temp
        history_copy.history = temp.detach().clone()  # use torch.clone() for tensor
        return history_copy

    @torch.no_grad()
    def reset(self, reset_mask = None):
        """
        mask: Bool tensor [B]
        Zero out entire history for envs that just terminated.
        """
        # global reset:
        if reset_mask is None:
            self.history = torch.full(
                (
                    self.config.batch_size, 
                    self.config.history_size,
                    self.config.observation_size
                ),
                fill_value = self.config.reset_value,
                dtype=self.dtype,
                device=self.device,
            )
            return

        # conditional reset:        
        if reset_mask.any():
            self.history[reset_mask] = self.config.reset_value

    def update(self, observation, reset_mask=None):
        """
        new_obs: [B, O]
        reset_mask: optional, Bool [B]; True = Reset
        """
        # 1. First reset terminated environments
        if reset_mask is not None:
            self.reset(reset_mask)

        # 2. Roll history to the left (drop oldest)
        # Equivalent to: history[:, 1:T] -> history[:, 0:T-1]
        # DO NOT KEEP GRADS FROM PREVIOUS FORWARD PASSES
        self.history = self.update_history_tensor(self.history)

        # 3. Insert new observations at the end
        self.history[:, -1] = observation

    def get_tensor(self):
        """
        Returns full history tensor: [B, H, O]
        """
        return self.history
    
    def get_tensor_vector(self):
        return torch.flatten(self.history, start_dim=1)

    def vector_to_tensor(self, history_vector:torch.Tensor):
        """
        Reshapese history vector [B,H*O] back to 3d tensor: [B, H, O]
        """
        batch_size = history_vector.shape[0]
        history_size = self.config.history_size
        history_tensor = history_vector.view(batch_size, history_size, -1)
        return history_tensor

    def update_history_tensor(self, history:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            updated = history.detach().clone().roll(shifts=-1, dims=1)
        return updated