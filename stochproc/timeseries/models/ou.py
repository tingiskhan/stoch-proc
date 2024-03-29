import torch
from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..linear import LinearModel


def _initial_kernel(kappa, gamma, sigma):
    return Normal(loc=gamma, scale=sigma / (2 * kappa).sqrt())


class OrnsteinUhlenbeck(LinearModel):
    r"""
    Implements the solved Ornstein-Uhlenbeck process, i.e. the solution to the SDE
        .. math::
            dX_t = \kappa (\gamma - X_t) dt + \sigma dW_t, \newline
            X_0 \sim \mathcal{N}(\gamma, \frac{\sigma}{\sqrt{2 \kappa}},

    where :math:`\kappa, \sigma \in \mathbb{R}_+^n`, and :math:`\gamma \in \mathbb{R}^n`.
    """

    def __init__(self, kappa: ParameterType, gamma: ParameterType, sigma: ParameterType, dt: float = 1.0):
        """
        Internal initializer for :class:`OrnsteinUhlenbeck`.

        Args:
            kappa: reversion parameter.
            gamma: mean parameter.
            sigma: volatility parameter.
            dt: the timestep to use.
            kwargs: see base.
        """

        kappa, gamma, sigma = broadcast_all(kappa, gamma, sigma)
        increment_distribution = Normal(torch.tensor(0.0, device=kappa.device), torch.tensor(1.0, device=kappa.device))
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

        # Parameter transform
        a = (-kappa * self._dt).exp()
        b = gamma * (1.0 - a)
        s = sigma / (2.0 * kappa).sqrt() * (1.0 - a.pow(2.0)).sqrt()

        super().__init__(
            (a, b, s),
            increment_distribution,
            initial_kernel=_initial_kernel,
            initial_parameters=(kappa, gamma, sigma),
        )

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        new = self._get_checked_instance(OrnsteinUhlenbeck)

        super(OrnsteinUhlenbeck, new).__init__(
            new_parameters["parameters"],
            self.increment_distribution,
            self._initial_kernel,
            new_parameters["initial_parameters"],
        )
        new._dt = self._dt

        return new
