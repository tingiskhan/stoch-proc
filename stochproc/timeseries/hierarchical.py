import torch

from .affine import AffineProcess
from .joint import AffineJointStochasticProcess, LowerCholeskyJointStochasticProcess, _multiplier


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
            >>> from pyro.distributions import Normal, LogNormal
            >>>
            >>> def mean_scale(x, kappa, theta, sigma):
            >>>     return kappa * (theta - x["sub"].value / kappa - x.value), sigma
            >>>
            >>> dt = 1.0
            >>> u = ts.models.OrnsteinUhlenbeck(0.01, 0.0, 0.01, dt=dt)
            >>>
            >>> inc_dist = dists.DistributionModule(Normal, loc=0.0, scale=sqrt(dt))
            >>> init_dist = dists.DistributionModule(LogNormal, loc=-2.0, scale=0.5)
            >>> hull_white = ts.AffineEulerMaruyama(mean_scale, (0.01, 0.5, 0.05), init_dist, inc_dist, dt).add_sub_process(u)
            >>>
            >>> x = hull_white.sample_states(500).get_path()
            >>> x.shape
            torch.Size([500, 2])

    .. _`Hull-White model`: https://en.wikipedia.org/wiki/Hull–White_model
    """

    def __init__(self, sub_process: AffineProcess, main_process: AffineProcess):
        """
        Initializes the :class:`AffineHierarchalProcess` class.

        Args:
            sub_process: child/sub process.
            main_process: main process.
        """

        super(AffineHierarchicalProcess, self).__init__(sub=sub_process, main=main_process)

    def mean_scale(self, x, parameters=None):
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


# TODO: Code duplication...
class LowerCholeskyHierarchicalProcess(LowerCholeskyJointStochasticProcess):
    r"""
    Similar to :class:`AffineHierarchicalProcess` but instead uses
    :class:`pyro.distributions.transforms.LowerCholeskyAffine`.
    """

    def __init__(self, sub_process: AffineProcess, main_process: AffineProcess):
        """
        Initializes the :class:`AffineHierarchalProcess` class.

        Args:
            sub_process: child/sub process.
            main_process: main process.
        """

        super(LowerCholeskyHierarchicalProcess, self).__init__(sub=sub_process, main=main_process)

    def mean_scale(self, x, parameters=None):
        sub_mean, sub_scale = self.sub_processes["sub"].mean_scale(x["sub"])

        main_state = x["main"]
        main_state["sub"] = x["sub"]
        main_mean, main_scale = self.sub_processes["main"].mean_scale(main_state)

        sub_numel = x["sub"].event_shape.numel()

        mean = (
            sub_mean.unsqueeze(-1) if sub_numel == 1 else sub_mean,
            main_mean,
        )

        eye = torch.eye(self.event_shape.numel(), device=x.value.device)

        scale = (
            _multiplier(sub_scale, eye[:sub_numel], self.sub_processes["sub"], x.batch_shape),
            _multiplier(main_scale, eye[sub_numel:], self.sub_processes["main"], x.batch_shape)
        )

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-2)
