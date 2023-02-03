import torch
from torch.distributions.utils import broadcast_all

from .ar import AR


class Seasonal(AR):
    r"""
    Defines a seasonal time series model, i.e. in which we have that
        .. math::
            s_{1, t + 1} = -\sum^{k}_{j=1} s_{j, t} + \epsilon_{t + 1}, \: \epsilon_t \sim \mathcal{N}(0, \sigma), \newline
            s_{j, t + 1} = s_{j-1, t}, j \geq 2

    """

    def __init__(self, period: int, sigma):
        """
        Internal initializer for :class:`Seasonal` model.

        Args:
            period: period to use for the seasonal model.
            sigma: standard deviation of the Guassian noise.
            initial_sigma: optional, initial standard deviation. If ``None`` uses ``sigma``.
        """

        sigma = broadcast_all(sigma)[0]

        alpha = torch.ones(sigma.shape + torch.Size([period]), device=sigma.device)
        super().__init__(torch.zeros_like(sigma), alpha, sigma, lags=period)
