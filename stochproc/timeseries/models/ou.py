import torch
from pyro.distributions import Normal

from ..affine import AffineProcess
from ...distributions import DistributionModule
from ...typing import ParameterType
from ...utils import enforce_named_parameter


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

    def __init__(
        self, kappa: ParameterType, gamma: ParameterType, sigma: ParameterType, dt: float = 1.0, **kwargs
    ):
        """
        Initializes the :class:`OrnsteinUhlenbeck` class.

        Args:
            kappa: reversion parameter.
            gamma: mean parameter.
            sigma: volatility parameter.
            dt: the timestep to use.
            kwargs: see base.
        """

        kappa, gamma, sigma = enforce_named_parameter(kappa=kappa, gamma=gamma, sigma=sigma)

        dist = DistributionModule(Normal, loc=0.0, scale=1.0)
        initial_dist = DistributionModule(init_builder, kappa=kappa, gamma=gamma, sigma=sigma)

        super().__init__(self._mean_scale, (kappa, gamma, sigma), initial_dist, dist, **kwargs)
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _mean_scale(self, x, k, g, s):
        loc = g + (x.values - g) * (-k * self._dt).exp()
        scale = s / (2 * k).sqrt() * (1 - (-2 * k * self._dt).exp()).sqrt()

        return loc, scale
