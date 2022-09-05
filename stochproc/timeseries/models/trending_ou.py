from torch.distributions.utils import broadcast_all
from pyro.distributions import Normal
import torch

from .ou import init_builder
from ..affine import AffineProcess
from ...distributions import DistributionModule
from ...typing import ParameterType


class TrendingOU(AffineProcess):
    r"""
    Implements the `Trending OU process`_.

    .. _`Trending OU process`: https://deanstreetlab.github.io/papers/papers/Statistical%20Methods/Trending%20Ornstein-Uhlenbeck%20Process%20and%20its%20Applications%20in%20Mathematical%20Finance.pdf
    """

    def __init__(
        self,
        kappa: ParameterType,
        gamma: ParameterType,
        sigma: ParameterType,
        v_0: ParameterType,
        dt: float = 1.0,
        **kwargs
    ):
        """
        Initializes the :class:`TrendingOU` object.

        Args:
            kappa: reversion parameter.
            gamma: mean parameter.
            sigma: volatility parameter.
            v_0: initial value.
            dt: time discretization step.
        """

        kappa, gamma, sigma, v_0 = broadcast_all(kappa, gamma, sigma, v_0)
        dist = DistributionModule(Normal, loc=0.0, scale=1.0)
        initial_dist = DistributionModule(init_builder, kappa=kappa, gamma=v_0, sigma=sigma)

        super().__init__(self._mean_scale, (kappa, gamma, v_0, sigma), initial_dist, dist, **kwargs)
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _mean_scale(self, x, k, g, v_0, s):
        d = (-k * self._dt).exp()
        loc = v_0 + g * (x.time_index + self._dt) + (x.values - g * x.time_index - v_0) * d
        scale = s / (2.0 * k).sqrt() * (1.0 - d.pow(2.0)).sqrt()

        return loc, scale
