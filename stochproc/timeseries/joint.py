import functools

import torch
from torch.distributions import Distribution
from .stochastic_process import StochasticProcess
from .affine import AffineProcess
from .state import TimeseriesState, JointState
from ..distributions import JointDistribution, DistributionModule


# class JointStochasticProcess(StochasticProcess):
#     """
#     A stochastic process comprising multiple separate stochastic processes by assuming independence between them. That
#     is, given :math:`n` stochastic processes :math:`\\{X^i_t\\}, i = 1, \\dots, n` we have
#         .. math::
#             p(x^1_{t+1}, \\dots, x^n_{t+1} \\mid x^1_t, \\dots, x^n_t) = \\prod^n_{i=1} p(x^i_{t+1} \\mid x^i_t)
#
#     Example:
#         In this example we'll construct a joint process of a random walk and an Ornstein-Uhlenbeck process.
#             >>> from pyfilter.timeseries import models as m, JointStochasticProcess
#             >>>
#             >>> ou = m.OrnsteinUhlenbeck(0.01, 0.0, 0.05, 1, 1.0)
#             >>> rw = m.RandomWalk(0.05)
#             >>>
#             >>> joint = JointStochasticProcess(ou=ou, rw=rw)
#             >>> x = joint.sample_path(1000)
#             >>> x.shape
#             torch.Size([1000, 2])
#     """
#
#     def __init__(self, **processes: StochasticProcess):
#         """
#         Initializes the ``JointStochasticProcess`` class.
#
#         Args:
#             processes: Key-worded processes, where the process will be registered as a module with key.
#         """
#
#         if any(not isinstance(v, StochasticProcess) for v in processes.values()):
#             raise Exception(f"All kwargs were not of type ``{StochasticProcess.__name__}``!")
#
#         joint_dist = JointDistribution(*(p.initial_dist for p in processes.values()))
#         self.indices = joint_dist.indices
#
#         super().__init__(DistributionModule(lambda **u: joint_dist))
#
#         self._proc_names = processes.keys()
#         for name, proc in processes.items():
#             self.add_module(name, proc)
#
#     def initial_sample(self, shape=None) -> JointState:
#         return JointState(
#             *(self._modules[name].initial_sample(shape) for name in self._proc_names), indices=self.indices
#         )
#
#     def build_density(self, x: JointState) -> Distribution:
#         return JointDistribution(
#             *(self._modules[name].build_density(x[i]) for i, name in enumerate(self._proc_names)), indices=self.indices
#         )


class AffineJointStochasticProcess(AffineProcess):
    """
    Similar to ``JointStochasticProcess`` but with the exception that all processes are of type ``AffineProcess``. This
    allows us to concatenate the mean and scale processes.
    """

    def __init__(self, **processes: AffineProcess):
        """
        Initializes the ``AffineJointStochasticProcess`` class.

        Args:
            processes: See base.
        """

        if not all(isinstance(p, AffineProcess) for p in processes.values()):
            raise ValueError(f"All processes must be of type '{AffineProcess.__name__}'!")

        super(AffineJointStochasticProcess, self).__init__(
            None,
            (),
            increment_dist=DistributionModule(self._inc_builder),
            initial_dist=DistributionModule(self._init_builder)
        )

        self.sub_processes = torch.nn.ModuleDict(processes)

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
