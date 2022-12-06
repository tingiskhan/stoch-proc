from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all
import torch

from ..affine import AffineProcess
from ..hierarchical import AffineHierarchicalProcess
from ...typing import ParameterType


def _mean_scale(x, s, lam):
    return x.value + lam * x["sub"].value, s


def initial_kernel(l_0, eps):
    return Normal(l_0, eps)


class SmoothLinearTrend(AffineHierarchicalProcess):
    """
    Implements a smooth linear trend. It's similar to :class:`stochproc.timeseries.models.LocalLinearTrend`, but we
    modify the level component a bit and get the following instead
        .. math::
            L_{t+1} = L_t + S_t.

    However, in order to avoid singularities when using variational inference we'll assume that instead of using a
    Delta distribution, we use a very low variance Gaussian distribution.
    """

    def __init__(self, trend_process: AffineProcess, l_0: ParameterType = 0.0, scaling: float = 1.0, eps: float = 1e-5):
        """
        Initializes the :class:`SmoothLinearTrend` class.

        Args:
            trend_process: model to use for modelling the trend component.
            l_0: initial level value.
            scaling: factor to apply to sub process.
        """

        l_0, scaling, eps = broadcast_all(l_0, scaling, eps)

        inc_dist = Normal(loc=torch.zeros_like(l_0), scale=torch.ones_like(l_0))
        level_process = AffineProcess(_mean_scale, inc_dist, (eps, scaling), initial_kernel=initial_kernel, initial_parameters=(l_0, eps))

        super().__init__(trend_process, level_process)
