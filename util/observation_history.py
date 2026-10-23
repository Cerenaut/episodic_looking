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

    Also keeps each row's first observation since its last reset (first, [B, O]): the history drops it once
    more than history_size observations have arrived, e.g. at the final transition of an episode exactly
    history_size steps long (the CIFAR setting: 8 and 8).
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
        history_copy.first = self.first.detach().clone()
        history_copy.has_first = self.has_first.clone()
        return history_copy

    def is_empty(self) -> torch.Tensor:
        """[B] bool: rows with no observation since their last reset."""
        return ~self.has_first

    def get_first(self) -> torch.Tensor:
        """[B, O] first observation since each row's last reset (zeros if none)."""
        return self.first

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
            self.first = torch.zeros(
                (self.config.batch_size, self.config.observation_size), dtype=self.dtype, device=self.device)
            self.has_first = torch.zeros(self.config.batch_size, dtype=torch.bool, device=self.device)
            return

        # conditional reset:        
        if reset_mask.any():
            self.history[reset_mask] = self.config.reset_value
            self.first[reset_mask] = 0
            self.has_first[reset_mask] = False

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

        # 4. First observation since the last reset
        with torch.no_grad():
            new = ~self.has_first
            if new.any():
                self.first[new] = observation.detach()[new]
                self.has_first |= new

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