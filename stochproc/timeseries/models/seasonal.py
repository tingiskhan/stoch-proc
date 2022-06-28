import torch
from stochproc.distributions import DistributionModule

from ..linear import LinearModel, ParameterType
from .ar import _build_init, _build_trans_dist


class Seasonal(LinearModel):
    r"""
    Defines a seasonal time series model, i.e. in which we have that
        .. math::
            s_t = -\sum^{k-1}_{j=1} s_{t-j} + \epsilon_t, \: \epsilon_t \sim \mathcal{N}(0, \sigma).
    """

    def __init__(self, period: int, sigma: ParameterType):
        """
        Initializes the :class:`Seasonal` model.

        Args:
            period: period to use for the seasonal model.
            sigma: standard deviation of the Guassian noise.
        """

        mat = torch.eye(period - 1, period)
        mat = torch.cat((-torch.ones((1, period)), mat), dim=0)

        inc_dist = DistributionModule(_build_trans_dist, loc=0.0, scale=1.0, lags=period)
        initial_dist = DistributionModule(_build_init, alpha=0.0, beta=1.0, sigma=sigma, lags=period)

        super(Seasonal, self).__init__(mat, sigma, increment_dist=inc_dist, initial_dist=initial_dist)

