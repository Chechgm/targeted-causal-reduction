from typing import Optional

import torch
from torch import nn

from .reduction_map import LinearCausalAbstraction
from ..causal_model import LowLevelCausalModel, HighLevelCausalModel


class CausalAbstraction(nn.Module):
    """
    A causal abstraction learns a mapping from a low-level causal model to a high-level causal model.

    Attributes
    ----------
    low_level : LowLevelCausalModel
        Low-level causal model.
    high_level : HighLevelCausalModel
        High-level causal model.
    tau_map_ground_truth : Optional[LinearCausalAbstraction]
        Ground truth mapping from low-level to high-level variables. This only exists for synthetic linear
        low-level causal models.
    omega_map_ground_truth : Optional[LinearCausalAbstraction]
        Ground truth mapping from low-level to high-level interventions. This only exists for synthetic linear
        low-level causal models.
    tau_map : LinearCausalAbstractionMap
        Mapping from low-level to high-level variables.
    omega_map : LinearCausalAbstractionMap
        Mapping from low-level to high-level interventions.
    """

    def __init__(
        self,
        low_level: LowLevelCausalModel,
        high_level: HighLevelCausalModel,
        tau_map_ground_truth: Optional[LinearCausalAbstraction] = None,
        omega_map_ground_truth: Optional[LinearCausalAbstraction] = None,
    ):
        super().__init__()
        self.low_level = low_level
        self.high_level = high_level
        self.tau_map_ground_truth = tau_map_ground_truth
        self.omega_map_ground_truth = omega_map_ground_truth

        self.omega_map = LinearCausalAbstraction(low_level.n_vars, high_level.n_vars)
        self.tau_map = LinearCausalAbstraction(low_level.n_vars, high_level.n_vars)

    def forward(
        self, x: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the reduction. It maps low-level variables `x` and interventions 's' to high-level causes 'z_hat'
        and interventions `r`, respectively, and then applies the high-level causal mechanism to predict the target.

        Parameters
        ----------
        x: torch.Tensor
            Low-level variables. Shape (batch_size, batch_size_sim, self.high_level.n_vars).
        s: torch.Tensor
            Low-level interventions. Shape (batch_size, batch_size_sim, self.high_level.n_vars).

        Returns
        -------
        z_hat: torch.Tensor
            High-level causes. Shape (batch_size, batch_size_sim, self.high_level.n_vars).
        r: torch.Tensor
            High-level interventions. Shape (batch_size, batch_size_sim, self.high_level.n_vars).
        """
        shape = (
            *x.shape[:-1],
            self.high_level.n_vars,
        )  # (batch_size, batch_size_sim, n_vars_high_level)
        z_hat = torch.zeros(shape, device=x.device)  # abstract variables
        r = torch.zeros_like(z)  # abstract interventions

        for i in range(self.high_level.n_vars):
            z_hat[..., i] = self.tau_map(x, idx=i).squeeze(-1)
            r[..., i] = self.omega_map(s, idx=i).squeeze(-1)

        return z_hat, r
