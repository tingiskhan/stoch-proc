from collections import OrderedDict
from functools import partial
import torch
from pyro.distributions import Distribution
from typing import Dict, Sequence
from contextlib import ExitStack, contextmanager


from ..distributions import JointDistribution
from .affine import AffineProcess
from .state import JointState, TimeseriesState
from .stochastic_process import StructuralStochasticProcess
from ..typing import ParameterType


def _initial_kernel(sub_processes, *_):
    return JointDistribution(*(sub_proc.initial_distribution for sub_proc in sub_processes.values()))


class _JointMixin(object):
    """
    Helper object for initializing joint models.
    """

    def __init__(self, **processes: AffineProcess):
        """
        Initializes the :class:`_InitializerMixin` class.

        Args:
            processes: sub processes to combine into a single affine process.
        """

        self.sub_processes = OrderedDict(
            {k: v for k, v in processes.items() if isinstance(v, StructuralStochasticProcess)}
        )
        self._initial_kernel = partial(_initial_kernel, sub_processes=self.sub_processes)
        self._event_shape = self.initial_distribution.event_shape

        self.parameters = ()
        self.initial_parameters = ()

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> JointState:
        return JointState(**{proc_name: proc.initial_sample(shape) for proc_name, proc in self.sub_processes.items()})

    def yield_parameters(self, filt=None):
        res = OrderedDict([])

        for k, v in self.sub_processes.items():
            for sk, sv in v.yield_parameters(filt).items():
                if sk not in res:
                    res[sk] = OrderedDict([])

                res[sk][k] = sv

        return res

    # NB: A tad bit hacky, but this should be possible to do using generics?
    def build_density(self, x) -> Distribution:
        return self._kernel(x, *self.parameters)

    @contextmanager
    def override_parameters(self, parameters: Dict[str, Sequence[ParameterType]]):
        try:
            t = (self.sub_processes[k].override_parameters(v) for k, v in parameters.items())
            with ExitStack() as stack:
                for cm in t:
                    stack.enter_context(cm)

                yield self
        finally:
            pass


class JointStochasticProcess(_JointMixin, StructuralStochasticProcess):
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
        Internal initializer for :class:`JointStochasticProcess`.

        Args:
            processes: sub processes to combine into a single affine process.
        """

        StructuralStochasticProcess.__init__(
            self,
            kernel=self._joint_kernel,
            parameters=(),
            initial_kernel=None,
            initial_parameters=(),
        )

        _JointMixin.__init__(self, **processes)

    def _joint_kernel(self, x: TimeseriesState) -> Distribution:
        return JointDistribution(*(sub_proc.build_density(x[k]) for k, sub_proc in self.sub_processes.items()))

    def expand(self, batch_shape):
        return JointStochasticProcess(**{k: v.expand(batch_shape) for k, v in self.sub_processes.items()})


class AffineJointStochasticProcess(_JointMixin, AffineProcess):
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
        Internal initializer for :class:`AffineJointStochasticProcess`.

        Args:
            processes: sub processes to combine into a single affine process.
        """

        msg = f"All processes must be of type '{AffineProcess.__name__}'!"
        assert all(issubclass(p.__class__, AffineProcess) for p in processes.values()), msg

        increment_distribution = JointDistribution(
            *(sub_proc.increment_distribution for sub_proc in processes.values())
        )

        AffineProcess.__init__(
            self,
            mean_scale=None,
            increment_distribution=increment_distribution,
            parameters=(),
            initial_kernel=None,
            initial_parameters=(),
        )

        _JointMixin.__init__(self, **processes)

        self._unsqueeze_mapper = {
            proc_name: partial(self._unsqueeze_wrapper, do_unsqueeze=proc.n_dim == 0)
            for proc_name, proc in self.sub_processes.items()
        }

    def mean_scale(self, x: TimeseriesState):
        mean = tuple()
        scale = tuple()

        for proc_name, proc in self.sub_processes.items():
            m, s = proc.mean_scale(x[proc_name])

            unsqueezer = self._unsqueeze_mapper[proc_name]

            mean += (unsqueezer(m),)
            scale += (unsqueezer(s),)

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)

    def expand(self, batch_shape):
        return AffineJointStochasticProcess(**{k: v.expand(batch_shape) for k, v in self.sub_processes.items()})

    @staticmethod
    def _unsqueeze_wrapper(u: torch.Tensor, do_unsqueeze: bool):
        if not do_unsqueeze:
            return u

        return u.unsqueeze(-1)


def joint_process(**processes: StructuralStochasticProcess) -> StructuralStochasticProcess:
    """
    Primitive for constructing joint stochastic processes.

    Args:
        **processes: processes to combine into one joint stochastic process.

    Returns:
        Returns a suitable joint stochastic process.
    """

    all_affine = all(isinstance(p, AffineProcess) for p in processes.values())

    if all_affine:
        return AffineJointStochasticProcess(**processes)

    return JointStochasticProcess(**processes)
