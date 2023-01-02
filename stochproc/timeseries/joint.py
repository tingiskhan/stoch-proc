import torch
from pyro.distributions import TransformedDistribution
from pyro.distributions import transforms as t
from torch.distributions import Distribution

from ..distributions import JointDistribution
from .affine import AffineProcess
from .chol_affine import LowerCholeskyAffineProcess
from .state import JointState, TimeseriesState
from .stochastic_process import StructuralStochasticProcess


# TODO: Perhaps unify with AffineJointStochasticProcess but I dislike multiple inheritance...
class JointStochasticProcess(StructuralStochasticProcess):
    r"""
    A stochastic process comprising multiple separate stochastic processes by assuming independence between them. That
    is, given :math:`n` stochastic processes :math:`\{ X^i_t \}, i = 1, \dots, n` we have
        .. math::
            p(x^1_{t+1}, \dots, x^n_{t+1} \mid x^1_t, \dots, x^n_t) = \prod^n_{i=1} p(x^i_{t+1} \mid x^i_t)

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
        Initializes the :class:`JointStochasticProcess` class.

        Args:
            processes: sub processes to combine into a single affine process.
        """

        parameters = sum((p.parameters for p in processes.values()), ())
        initial_parameters = sum((p.initial_parameters for p in processes.values()), ())

        super().__init__(
            kernel=self.kernel,
            parameters=set(parameters),
            initial_kernel=None,
            initial_parameters=set(initial_parameters),
        )

        self.sub_processes = {k: v for k, v in processes.items() if isinstance(v, StructuralStochasticProcess)}
        self._initial_kernel = self.initial_kernel
        self._event_shape = self.initial_distribution.event_shape

    def initial_kernel(self, *args):
        return JointDistribution(*(sub_proc.initial_distribution for sub_proc in self.sub_processes.values()))

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> JointState:
        return JointState(**{proc_name: proc.initial_sample(shape) for proc_name, proc in self.sub_processes.items()})

    def kernel(self, x: TimeseriesState, *args) -> Distribution:
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

        increment_distribution = JointDistribution(
            *(sub_proc.increment_distribution for sub_proc in processes.values())
        )

        # TODO: Code duplication... >>>
        parameters = sum((p.parameters for p in processes.values()), ())
        initial_parameters = sum((p.initial_parameters for p in processes.values()), ())

        super().__init__(
            mean_scale=None,
            increment_distribution=increment_distribution,
            parameters=parameters,
            initial_kernel=None,
            initial_parameters=initial_parameters,
        )

        self.sub_processes = {k: v for k, v in processes.items() if isinstance(v, StructuralStochasticProcess)}
        self._initial_kernel = self.initial_kernel
        self._event_shape = self.initial_distribution.event_shape

    def initial_kernel(self, *args):
        return JointDistribution(*(sub_proc.initial_distribution for sub_proc in self.sub_processes.values()))

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> JointState:
        return JointState(**{proc_name: proc.initial_sample(shape) for proc_name, proc in self.sub_processes.items()})

    # >>>

    def mean_scale(self, x: TimeseriesState, parameters=None):
        mean = tuple()
        scale = tuple()

        for i, (proc_name, proc) in enumerate(self.sub_processes.items()):
            overrides = parameters[i] if parameters is not None else None
            m, s = proc.mean_scale(x[proc_name], overrides)

            mean += (m.unsqueeze(-1) if proc.n_dim == 0 else m,)
            scale += (s.unsqueeze(-1) if proc.n_dim == 0 else s,)

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)


def _multiplier(
    s: torch.Tensor, eye: torch.Tensor, proc: StructuralStochasticProcess, batch_shape: torch.Size
) -> torch.Tensor:
    """
    Helper method for performing multiplying operation.

    Args:
        s: scale to use.
        proc: process to use.
    """

    if isinstance(proc, LowerCholeskyAffineProcess):
        res = s @ eye
    else:
        res = eye * (s.unsqueeze(-1) if proc.event_shape.numel() > 1 else s.view(*s.shape, 1, 1))

    return torch.broadcast_to(res, batch_shape + eye.shape)


class LowerCholeskyJointStochasticProcess(AffineJointStochasticProcess):
    r"""
    Similar to :class:`AffineJointStochasticProcess` but instead uses the
    :class:`pyro.distributions.transforms.LowerCholeskyAffine` of
    :class:`pyro.distributions.transforms.AffineTransform`.
    """

    def mean_scale(self, x: TimeseriesState, parameters=None):
        mean = tuple()
        scale = tuple()

        eye = torch.eye(self.event_shape.numel(), device=x.value.device)

        left = 0
        for i, (proc_name, proc) in enumerate(self.sub_processes.items()):
            overrides = parameters[i] if parameters is not None else None
            m, s = proc.mean_scale(x[proc_name], overrides)

            mean += (m.unsqueeze(-1) if proc.n_dim == 0 else m,)

            numel = proc.event_shape.numel()
            eye_slice = eye[left : left + numel]

            scale += (_multiplier(s, eye_slice, proc, x.batch_shape),)
            left += numel

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-2)

    # NB: Code duplication...
    def kernel(self, x, *args):
        loc, scale = self.mean_scale(x)

        return TransformedDistribution(self.increment_distribution, t.LowerCholeskyAffine(loc, scale))


def joint_process(**processes: StructuralStochasticProcess) -> StructuralStochasticProcess:
    """
    Primitive for constructing joint stochastic processes.

    Args:
        **processes: processes to combine into one joint stochastic process.

    Returns:
        Returns a suitable joint stochastic process.
    """

    chol_procs = (LowerCholeskyAffineProcess, LowerCholeskyJointStochasticProcess)
    all_affine = all(isinstance(p, AffineProcess) for p in processes.values())

    if any(isinstance(p, chol_procs) for p in processes.values()) and all_affine:
        return LowerCholeskyJointStochasticProcess(**processes)

    if all_affine:
        return AffineJointStochasticProcess(**processes)

    return JointStochasticProcess(**processes)
