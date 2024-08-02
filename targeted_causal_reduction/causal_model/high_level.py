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
    linear_macro_solution : nn.Module
        The linear layer to model the causal mechanism. This is equivalent to (I-A)^(-1)

    mu_z, sigma_z and the weights of the linear_macro_scm are learnable parameters.
    """

    def __init__(self, n_vars: int) -> None:
        super().__init__()
        self.n_vars = n_vars

        # TODO: in our case we don't need to learn these parameters
        #   but only the covariance of the noise variable Cov(N_hat),
        #   so we might be able to abstract the parameters that need 
        #   to be learned at the macro level.
        # register parameters for mu_z, sigma_z
        self.mu_N = nn.Parameter(torch.zeros(n_vars))
        self.cov_N = nn.Parameter(torch.ones(n_vars))

        self.linear_macro_solution = nn.Sequential(nn.Linear(n_vars, 1))

    def causal_mechanism(self, N):
        return self.linear_macro_solution(N)
