from math import sqrt

import torch
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import AbsTransform
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..diffusion import AffineEulerMaruyama
from .ou import initial_kernel as ou_builder


def _f(x, k, g, s):
    return k * (g - x.value) * x.value, s * x.value


def _initial_kernel(kappa, gamma, sigma):
    dist = ou_builder(kappa, gamma, sigma)

    return TransformedDistribution(dist, AbsTransform())


class Verhulst(AffineEulerMaruyama):
    r"""
    Implements a discretized Verhulst SDE with the following dynamics
        .. math::
            dX_t = \kappa (\gamma - X_t)X_t dt + \sigma X_t dW_t, \newlin
            X_0 \sim \left | \mathcal{N}(x_0, \frac{\sigma}{\sqrt{2\kappa}} \right |,

    where :math:`\kappa, \gamma, \sigma > 0`.
    """

    def __init__(self, kappa: ParameterType, gamma: ParameterType, sigma: ParameterType, dt):
        r"""
        Internal initializer for :class:`Verhulst`.

        Args:
            reversion: :math:`\kappa`.
            mean: :math:`\gamma`.
            vol: :math:`\sigma`.
            kwargs: see base.
        """

        kappa, gamma, sigma = broadcast_all(kappa, gamma, sigma)
        increment_distribution = Normal(
            torch.tensor(0.0, device=kappa.device), torch.tensor(sqrt(dt), device=kappa.device)
        )

        super().__init__(_f, (kappa, gamma, sigma), increment_distribution, dt, _initial_kernel)
