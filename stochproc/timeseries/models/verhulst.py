from math import sqrt

from torch.distributions.utils import broadcast_all
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import AbsTransform

from .ou import init_builder as ou_builder
from ..diffusion import AffineEulerMaruyama
from ...distributions import DistributionModule
from ...typing import ParameterType


def _f(x, k, g, s):
    return k * (g - x.values) * x.values, s * x.values


def _init_builder(kappa, gamma, sigma):
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

    def __init__(self, kappa: ParameterType, gamma: ParameterType, sigma: ParameterType, dt, **kwargs):
        """
        Initializes the :class:`Verhulst` class.

        Args:
            reversion: :math:`\\kappa`.
            mean: :math:`\\gamma`.
            vol: :math:`\\sigma`.
            kwargs: see base.
        """

        kappa, gamma, sigma = broadcast_all(kappa, gamma, sigma)

        super().__init__(
            _f,
            (kappa, gamma, sigma),
            DistributionModule(_init_builder, kappa=kappa, gamma=gamma, sigma=sigma),
            DistributionModule(Normal, loc=0.0, scale=sqrt(dt)),
            dt=dt,
            **kwargs
        )
