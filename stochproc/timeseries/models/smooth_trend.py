import torch
from pyro.distributions import Normal, Delta
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..affine import AffineProcess
from ..hierarchical import AffineHierarchicalProcess


def _mean_scale_0d(x, s, lam):
    return x.value + lam * x["sub"].value, s


def _mean_scale_1d(x, s, lam):
    return x.value + (lam.unsqueeze(-2) @ x["sub"].value.unsqueeze(-1)).reshape(x.value.shape), s


def initial_kernel(l_0):
    return Delta(l_0)


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
        Internal initializer for :class:`SmoothLinearTrend`.

        Args:
            trend_process: model to use for modelling the trend component.
            l_0: initial level value.
            scaling: factor to apply to sub process.
        """

        if trend_process.n_dim == 0:
            l_0, eps, scaling = broadcast_all(l_0, eps, scaling)
            mean_scale = _mean_scale_0d
        else:
            l_0, eps = broadcast_all(l_0, eps)
            (scaling,) = broadcast_all(scaling)
            scaling = scaling.expand(l_0.shape + scaling.shape)

            if scaling.shape[-1:] != trend_process.event_shape:
                raise Exception("Shapes not congruent!")

            mean_scale = _mean_scale_1d

        inc_dist = Normal(torch.tensor(0.0, device=l_0.device), torch.tensor(1.0, device=l_0.device))
        level_process = AffineProcess(
            mean_scale, (eps, scaling), inc_dist, initial_kernel=initial_kernel, initial_parameters=(l_0,)
        )

        super().__init__(trend_process, level_process)

    def expand(self, batch_shape):
        new = self._get_checked_instance(SmoothLinearTrend)
        super(AffineHierarchicalProcess, new).__init__(
            **{k: v.expand(batch_shape) for k, v in self.sub_processes.items()}
        )

        return new
