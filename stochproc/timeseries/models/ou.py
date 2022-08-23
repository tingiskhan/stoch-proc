import torch
from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all

from ..affine import AffineProcess
from ...distributions import DistributionModule
from ...typing import ParameterType


def init_builder(kappa, gamma, sigma):
    return Normal(loc=gamma, scale=sigma / (2 * kappa).sqrt())


# TODO: Should perhaps inherit from StochasticDifferentialEquation?
class OrnsteinUhlenbeck(AffineProcess):
    r"""
    Implements the solved Ornstein-Uhlenbeck process, i.e. the solution to the SDE
        .. math::
            dX_t = \kappa (\gamma - X_t) dt + \sigma dW_t, \newline
            X_0 \sim \mathcal{N}(\gamma, \frac{\sigma}{\sqrt{2 \kappa}},

    where :math:`\kappa, \sigma \in \mathbb{R}_+^n`, and :math:`\gamma \in \mathbb{R}^n`.
    """

    def __init__(self, kappa: ParameterType, gamma: ParameterType, sigma: ParameterType, dt: float = 1.0, **kwargs):
        """
        Initializes the :class:`OrnsteinUhlenbeck` class.

        Args:
            kappa: reversion parameter.
            gamma: mean parameter.
            sigma: volatility parameter.
            dt: the timestep to use.
            kwargs: see base.
        """

        kappa, gamma, sigma = broadcast_all(kappa, gamma, sigma)

        dist = DistributionModule(Normal, loc=0.0, scale=1.0)
        initial_dist = DistributionModule(init_builder, kappa=kappa, gamma=gamma, sigma=sigma)

        super().__init__(self._mean_scale, (kappa, gamma, sigma), initial_dist, dist, **kwargs)
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _mean_scale(self, x, k, g, s):
        d = (-k * self._dt).exp()
        loc = g + (x.values - g) * d
        scale = s / (2.0 * k).sqrt() * (1.0 - d.pow(2.0)).sqrt()

        return loc, scale
