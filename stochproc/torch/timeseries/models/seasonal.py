import torch
from torch.distributions.utils import broadcast_all
from stochproc.torch.distributions import DistributionModule

from .ar import _build_init, _build_trans_dist
from ..linear import LinearModel, ParameterType




class Seasonal(LinearModel):
    r"""
    Defines a seasonal time series model, i.e. in which we have that
        .. math::
            s_{1, t + 1} = -\sum^{k}_{j=1} s_{j, t} + \epsilon_{t + 1}, \: \epsilon_t \sim \mathcal{N}(0, \sigma), \newline
            s_{j, t + 1} = s_{j-1, t}, j \geq 2

    """

    def __init__(self, period: int, sigma: ParameterType, initial_sigma: ParameterType = None):
        """
        Initializes the :class:`Seasonal` model.

        Args:
            period: period to use for the seasonal model.
            sigma: standard deviation of the Guassian noise.
            initial_sigma: optional, initial standard deviation. If ``None`` uses ``sigma``.
        """

        sigma = broadcast_all(sigma)[0]

        mat = torch.eye(period - 1, period, device=sigma.device)
        mat = torch.cat((-torch.ones((1, period), device=sigma.device), mat), dim=0)

        inc_dist = DistributionModule(_build_trans_dist, loc=0.0, scale=1.0, lags=period)
        initial_dist = DistributionModule(
            _build_init, alpha=0.0, beta=1.0, sigma=sigma if initial_sigma is None else initial_sigma, lags=period
        )

        super(Seasonal, self).__init__(mat, sigma, increment_dist=inc_dist, initial_dist=initial_dist)