import torch
from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all

from ..linear import LinearModel
from ...distributions import DistributionModule
from ...typing import ParameterType


class LocalLinearTrend(LinearModel):
    r"""
    Implements a Local Linear Trend model, i.e. a model with the following dynamics
        .. math::
            L_{t+1} = L_t + S_t + \sigma_l W_{t+1}, \newline
            S_{t+1} = S_t + \sigma_s V_{t+1},

    where :math:`\sigma_i > 0``, and :math:`W_t, V_t` are two independent zero mean and unit variance Gaussians.
    """

    def __init__(self, sigma: ParameterType, initial_mean: ParameterType = torch.zeros(2), **kwargs):
        r"""
        Initializes the :class:`LocalLinearTrend` class.

        Args:
            sigma: the vector :math:`[ \sigma_s, \sigma_l ]`.
            initial_mean: the initial mean.
            kwargs: see base.
        """

        sigma, initial_mean = broadcast_all(sigma, initial_mean)

        increment_dist = DistributionModule(Normal, loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)
        initial_dist = DistributionModule(Normal, loc=initial_mean, scale=sigma).to_event(1)
        a = torch.tensor([[1.0, 0.0], [1.0, 1.0]], device=sigma.device)

        super().__init__(a, sigma, increment_dist=increment_dist, initial_dist=initial_dist, **kwargs)
