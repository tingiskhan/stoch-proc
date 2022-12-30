import torch
from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..linear import LinearModel


def initial_kernel(kappa, gamma, sigma):
    return Normal(loc=gamma, scale=sigma / (2 * kappa).sqrt())


# TODO: Should perhaps inherit from StochasticDifferentialEquation?
class OrnsteinUhlenbeck(LinearModel):
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
        increment_distribution = Normal(torch.tensor(0.0, device=kappa.device), torch.tensor(1.0, device=kappa.device))

        super().__init__(
            kappa,
            sigma,
            increment_distribution,
            initial_kernel=initial_kernel,
            parameter_transform=self._parameter_transform,
            b=gamma,
            **kwargs
        )

        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _parameter_transform(self, k, g, s):
        a = (-k * self._dt).exp()
        b = g * (1.0 - a)
        s = s / (2.0 * k).sqrt() * (1.0 - a.pow(2.0)).sqrt()

        return a, b, s
