import torch

from stochproc.timeseries.stochastic_process import StructuralStochasticProcess

from ..distributions import JointDistribution
from .affine import AffineProcess
from .joint import AffineJointStochasticProcess, JointStochasticProcess


class AffineHierarchicalProcess(AffineJointStochasticProcess):
    r"""
    Defines a "hierarchal" affine process, by which we mean a stochastic process that comprises two sub processes,
    where one is completely independent of the other, whereas the other is conditionally independent; in theory similar
    to :class:`stochproc.timeseries.StateSpaceModel`.

    To clarify, assume that you have the stochastic processes :math:`\{ X_t \}` and :math:`\{ Y_t \}`, and that
    :math:`X` is independent of :math:`Y`, but :math:`Y` is conditionally independent of :math:`X`.

    Example:
        One example is the two factor `Hull-White model`_, which in code is defined as (with arbitrary parameters)
            >>> from stochproc import timeseries as ts, distributions as dists
            >>> from math import sqrt
            >>> import torch
            >>> from pyro.distributions import Normal, LogNormal
            >>>
            >>> def mean_scale(x, kappa, theta, sigma):
            >>>     return kappa * (theta - x["sub"].value / kappa - x.value), sigma
            >>>
            >>> def initial_kernel(kappa, theta, sigma):
            >>>     return Normal(torch.zeros_like(kappa), torch.ones_like(kappa))
            >>>
            >>> dt = 1.0
            >>> u = ts.models.OrnsteinUhlenbeck(0.01, 0.0, 0.01, dt=dt)
            >>>
            >>> inc_dist = Normal(loc=0.0, scale=sqrt(dt))
            >>> hull_white = ts.AffineEulerMaruyama(mean_scale, (0.01, 0.5, 0.05), inc_dist, initial_kernel=initial_kernel, dt=dt).add_sub_process(u)
            >>>
            >>> x = hull_white.sample_states(500).get_path()
            >>> x.shape
            torch.Size([500, 2])

    .. _`Hull-White model`: https://en.wikipedia.org/wiki/Hull–White_model
    """

    def __init__(self, sub_process: AffineProcess, main_process: AffineProcess):
        """
        Internal initializer for :class:`AffineHierarchicalProcess`.

        Args:
            sub_process: child/sub process.
            main_process: main process.
        """

        super().__init__(sub=sub_process, main=main_process)

    def mean_scale(self, x):
        sub_mean, sub_scale = self.sub_processes["sub"].mean_scale(x["sub"])

        main_state = x["main"]
        main_state["sub"] = x["sub"]
        main_mean, main_scale = self.sub_processes["main"].mean_scale(main_state)

        sub_unsqueeze = x["sub"].event_shape.numel() == 1
        main_unsqueeze = x["main"].event_shape.numel() == 1

        mean = (
            sub_mean.unsqueeze(-1) if sub_unsqueeze else sub_mean,
            main_mean.unsqueeze(-1) if main_unsqueeze else main_mean,
        )

        scale = (
            sub_scale.unsqueeze(-1) if sub_unsqueeze else sub_scale,
            main_scale.unsqueeze(-1) if main_unsqueeze else main_scale,
        )

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)

    def expand(self, batch_shape):
        return AffineHierarchicalProcess(**{k: v.expand(batch_shape) for k, v in self.sub_processes.items()})


class HierarchicalProcess(JointStochasticProcess):
    """
    See :class:`AffineHierarchicalProcess`.
    """

    def __init__(self, sub_process: StructuralStochasticProcess, main_process: StructuralStochasticProcess):
        """
        Internal initializer for :class:`HierarchicalProcess`.

        Args:
            sub_process (StructuralStochasticProcess): child/sub process.
            main_process (StructuralStochasticProcess): main process.
        """

        super().__init__(sub=sub_process, main=main_process)

    # TODO: COnsider saving indices and re-using
    def _joint_kernel(self, x):
        sub_dist = self.sub_processes["sub"].build_density(x["sub"])

        main_state = x["main"]
        main_state["sub"] = x["sub"]
        main_dist = self.sub_processes["main"].build_density(main_state)

        return JointDistribution(sub_dist, main_dist)
