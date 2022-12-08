import torch
from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..linear import LinearModel


def initial_kernel(x_0, sigma):
    return Normal(x_0, sigma).to_event(1)


class LocalLinearTrend(LinearModel):
    r"""
    Implements a Local Linear Trend model, i.e. a model with the following dynamics
        .. math::
            L_{t+1} = L_t + S_t + \sigma_l W_{t+1}, \newline
            S_{t+1} = S_t + \sigma_s V_{t+1},

    where :math:`\sigma_i > 0``, and :math:`W_t, V_t` are two independent zero mean and unit variance Gaussians.
    """

    def __init__(self, sigma: ParameterType, initial_mean: ParameterType = None):
        r"""
        Initializes the :class:`LocalLinearTrend` class.

        Args:
            sigma: the vector :math:`[ \sigma_s, \sigma_l ]`.
            initial_mean: the initial mean.
            kwargs: see base.
        """

        sigma = broadcast_all(sigma)[0]

        if not initial_mean:
            initial_mean = torch.zeros_like(sigma)
        else:
            sigma, initial_mean = broadcast_all(sigma, initial_mean)

        increment_dist = (
            Normal(loc=torch.tensor(0.0, device=sigma.device), scale=torch.tensor(1.0, device=sigma.device))
            .expand(torch.Size([2]))
            .to_event(1)
        )

        a = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=sigma.device)

        super().__init__(
            a,
            sigma,
            increment_distribution=increment_dist,
            initial_kernel=initial_kernel,
            initial_parameters=(initial_mean, sigma),
        )
