from math import pi

from .cyclical import CyclicalProcess
from ..utils import coerce_tensors
from ...typing import ParameterType


class HarmonicProcess(CyclicalProcess):
    r"""
    Implements a harmonic timeseries process of the form
        .. math::
            \gamma_{t + 1} = \gamma \cos{ \lambda } + \gamma^*\sin{ \lambda } + \sigma \nu_{t + 1}, \newline
            \gamma^*_{t + 1} = -\gamma \sin { \lambda } + \gamma^* \cos{ \lambda } + \sigma^* \nu^*_{t + 1}.
        
    See `statsmodels`_.

    .. _`statsmodels`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html#statsmodels.tsa.statespace.structural.UnobservedComponents
    """

    def __init__(self, s: int, sigma: ParameterType, x_0: ParameterType = None, j: int = 1):
        """
        Internal initializer for :class:`HarmonicProcess`.

        Args:
            s (int): number of seasons.
            sigma (ParameterType): see :class:`Cyclical`.
            x_0 (ParameterType): see :class:`Cyclical`.
            j (int): "index" of harmonic process.
        """

        rho, lamda, sigma = coerce_tensors(1.0, 2.0 * pi * j / s, sigma)
        super().__init__(rho, lamda, sigma, x_0)
