from torch.distributions.utils import broadcast_all
from pyro.distributions import Normal
import torch

from .ou import initial_kernel
from ..affine import AffineProcess
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
        increment_distribution = Normal(torch.zeros_like(kappa), torch.ones_like(kappa))

        super().__init__(self._mean_scale, increment_distribution, (kappa, gamma, v_0, sigma), initial_kernel, initial_parameters=(kappa, v_0, sigma))
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _mean_scale(self, x, k, g, v_0, s):
        d = (-k * self._dt).exp()
        loc = v_0 + g * ((x.time_index + 1.0) * self._dt) + (x.value - g * x.time_index * self._dt - v_0) * d
        scale = s / (2.0 * k).sqrt() * (1.0 - d.pow(2.0)).sqrt()

        return loc, scale
