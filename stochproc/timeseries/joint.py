import functools

import torch
from torch.distributions import Distribution
from pyro.distributions import TransformedDistribution, transforms as t

from .stochastic_process import StructuralStochasticProcess
from .affine import AffineProcess
from .state import TimeseriesState, JointState
from .chol_affine import LowerCholeskyAffineProcess
from ..distributions import JointDistribution, DistributionModule


# TODO: Perhaps unify with AffineJointStochasticProcess but I dislike multiple inheritance...
class JointStochasticProcess(StructuralStochasticProcess):
    """
    A stochastic process comprising multiple separate stochastic processes by assuming independence between them. That
    is, given :math:`n` stochastic processes :math:`\\{X^i_t\\}, i = 1, \\dots, n` we have
        .. math::
            p(x^1_{t+1}, \\dots, x^n_{t+1} \\mid x^1_t, \\dots, x^n_t) = \\prod^n_{i=1} p(x^i_{t+1} \\mid x^i_t)

    Example:
        In this example we'll construct a joint process of a random walk and an Ornstein-Uhlenbeck process.
            >>> from stochproc.timeseries import models as m, JointStochasticProcess
            >>>
            >>> ou = m.OrnsteinUhlenbeck(0.01, 0.0, 0.05, 1, 1.0)
            >>> rw = m.RandomWalk(0.05)
            >>>
            >>> joint = JointStochasticProcess(ou=ou, rw=rw)
            >>> x = joint.sample_path(1000)
            >>> x.shape
            torch.Size([1000, 2])
    """

    def __init__(self, **processes: AffineProcess):
        """
        Initializes the :class:`AffineJointStochasticProcess` class.

        Args:
            processes: sub processes to combine into a single affine process.
        """

        super(JointStochasticProcess, self).__init__(parameters=(), initial_dist=None)

        self.sub_processes = torch.nn.ModuleDict(processes)

        self._initial_dist = DistributionModule(self._init_builder)
        self._event_shape = self.initial_dist.event_shape

    def _init_builder(self, **kwargs):
        return JointDistribution(*(sub_proc.initial_dist for sub_proc in self.sub_processes.values()))

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> JointState:
        return JointState(**{proc_name: proc.initial_sample(shape) for proc_name, proc in self.sub_processes.items()})

    def functional_parameters(self, **kwargs):
        return tuple((proc.functional_parameters(**kwargs) for proc in self.sub_processes.values()))

    def build_density(self, x: TimeseriesState) -> Distribution:
        return JointDistribution(*(sub_proc.build_density(x[k]) for k, sub_proc in self.sub_processes.items()))


class AffineJointStochasticProcess(AffineProcess):
    r"""
    Implements an affine joint stochastic process, i.e. a stochastic process comprising several conditionally
    independent affine sub processes. That is, given :math:`n` stochastic processes :math:`\{ X^i_t \}, i = 1, \dots, n`
    we have
         .. math::
             p(x^1_{t+1}, \dots, x^n_{t+1} \mid x^1_t, \dots, x^n_t) = \prod^n_{i=1} p( x^i_{t+1} \mid x^i_t ),

    where every sub process :math:`X^i` is of affine nature.
    """

    def __init__(self, **processes: AffineProcess):
        """
        Initializes the :class:`AffineJointStochasticProcess` class.

        Args:
            processes: sub processes to combine into a single affine process.
        """

        msg = f"All processes must be of type '{AffineProcess.__name__}'!"
        assert all(issubclass(p.__class__, AffineProcess) for p in processes.values()), msg

        super(AffineJointStochasticProcess, self).__init__(
            None, (), increment_dist=DistributionModule(self._inc_builder), initial_dist=None,
        )

        self.sub_processes = torch.nn.ModuleDict(processes)

        self._initial_dist = DistributionModule(self._init_builder)
        self._event_shape = self.initial_dist.event_shape

    def _inc_builder(self, **kwargs):
        return JointDistribution(*(sub_proc.increment_dist() for sub_proc in self.sub_processes.values()))

    def _init_builder(self, **kwargs):
        return JointDistribution(*(sub_proc.initial_dist for sub_proc in self.sub_processes.values()))

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> JointState:
        return JointState(**{proc_name: proc.initial_sample(shape) for proc_name, proc in self.sub_processes.items()})

    def mean_scale(self, x: TimeseriesState, parameters=None):
        mean = tuple()
        scale = tuple()

        for i, (proc_name, proc) in enumerate(self.sub_processes.items()):
            overrides = parameters[i] if parameters is not None else None
            m, s = proc.mean_scale(x[proc_name], overrides)

            mean += (m.unsqueeze(-1) if proc.n_dim == 0 else m,)
            scale += (s.unsqueeze(-1) if proc.n_dim == 0 else s,)

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)

    # TODO: Should perhaps return a flat list which is split later on, but I think this is better
    def functional_parameters(self, **kwargs):
        return tuple((proc.functional_parameters(**kwargs) for proc in self.sub_processes.values()))


def _multiplier(s: torch.Tensor, eye: torch.Tensor, proc: StructuralStochasticProcess) -> torch.Tensor:
    """
    Helper method for performing multiplying operation.

    Args:
        s: scale to use.
        proc: process to use.
    """

    if isinstance(proc, LowerCholeskyAffineProcess):
        return s @ eye

    return eye * (s if proc.event_shape.numel() == 0 else s.unsqueeze(-1))


class LowerCholeskyJointStochasticProcess(AffineJointStochasticProcess):
    r"""
    Similar to :class:`AffineJointStochasticProcess` but instead uses the
    :class:`pyro.distributions.transforms.LowerCholeskyAffine`.
    """

    def mean_scale(self, x: TimeseriesState, parameters=None):
        mean = tuple()
        scale = tuple()

        eye = torch.eye(self.event_shape.numel(), device=x.values.device)

        left = 0
        for i, (proc_name, proc) in enumerate(self.sub_processes.items()):
            overrides = parameters[i] if parameters is not None else None
            m, s = proc.mean_scale(x[proc_name], overrides)

            mean += (m.unsqueeze(-1) if proc.n_dim == 0 else m,)

            numel = proc.event_shape.numel()
            eye_slice = eye[left:left + numel]

            scale += (_multiplier(s, eye_slice, proc),)
            left += numel

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-2)

    # NB: Code duplication...
    def build_density(self, x):
        loc, scale = self.mean_scale(x)

        return TransformedDistribution(self.increment_dist(), t.LowerCholeskyAffine(loc, scale))


def joint_process(**processes: StructuralStochasticProcess) -> StructuralStochasticProcess:
    """
    Primitive for constructing joint stochastic processes.

    Args:
        **processes: processes to combine into one joint stochastic process.

    Returns:
        Returns a suitable joint stochastic process.
    """

