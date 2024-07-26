import torch
from torch import nn


class HighLevelCausalModel(nn.Module):
    """
    Linear Gaussian high-level causal model with `n_vars` macro variables 'z'.

    Attributes
    ----------
    n_vars : int
        The number of variables in the causal model; i.e., the number of macro variables.
    mu_z : nn.Parameter
        The mean of the unintervened macro variables 'z' of shape (n_vars,).
    sigma_z : nn.Parameter
        The standard deviation of the unintervened macro variables 'z' of shape (n_vars,).
    linear_macro_scm : nn.Module
        The linear layer to model the causal mechanism. This is equivalent to (I-A)^(-1)

    mu_z, sigma_z and the weights of the linear_macro_scm are learnable parameters.
    """

    def __init__(self, n_vars: int) -> None:
        super().__init__()
        self.n_vars = n_vars

        # register parameters for mu_z, sigma_z
        self.mu_z = nn.Parameter(torch.zeros(n_vars))
        self.sigma_z = nn.Parameter(torch.ones(n_vars))
        # self.macro_model = nn.Parameter(torch.ones(1)) # TODO: ask Armin

        self.linear_macro_scm = nn.Sequential(nn.Linear(n_vars, 1))

    def causal_mechanism(self, N):
        return self.linear_macro_scm(N)
