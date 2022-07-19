from pyro.distributions import Normal

from ..affine import AffineProcess
from ..hierarchical import AffineHierarchicalProcess
from ...distributions import DistributionModule
from ...typing import ParameterType


def _mean_scale(x, s):
    return x.values + x["sub"].values, s


class SmoothLinearTrend(AffineHierarchicalProcess):
    """
    Implements a smooth linear trend. It's similar to :class:`stochproc.timeseries.models.LocalLinearTrend`, but we
    modify the level component a bit and get the following instead
        .. math::
            L_{t+1} = L_t + S_t.

    However, in order to avoid singularities when using variational inference we'll assume that instead of using a
    Delta distribution, we use a very low variance Gaussian distribution.
    """

    def __init__(self, trend_process: AffineProcess, l_0: ParameterType = 0.0):
        """
        Initializes the :class:`SmoothLinearTrend` class.

        Args:
            trend_process: model to use for modelling the trend component.
            l_0: initial level value.
        """
        scale = 1e-5

        init_dist = DistributionModule(Normal, loc=l_0, scale=scale)
        inc_dist = DistributionModule(Normal, loc=0.0, scale=1.0)
        level_process = AffineProcess(_mean_scale, (scale,), init_dist, inc_dist)

        super().__init__(trend_process, level_process)