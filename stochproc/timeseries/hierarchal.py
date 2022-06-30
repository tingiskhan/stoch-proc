import torch

from . import JointState
from .affine import AffineProcess
from ..distributions import JointDistribution, DistributionModule


class AffineHierarchalProcess(AffineProcess):
    r"""
    Defines a "hierarchal" affine process, by which we mean a process that ... TODO

    """

    def __init__(self, sub_process: AffineProcess, mean_scale, parameters, initial_dist, increment_dist, **kwargs):
        """
        Initializes the :class:`AffineHierarchalProcess` class.

        Args:
            sub_process: child/sub process.
        """

        super().__init__(mean_scale, parameters, None, DistributionModule(self._inc_builder), **kwargs)
        self._initial_dist = initial_dist
        self._increment_dist = increment_dist

        self.sub_process = sub_process
        self._event_shape = self.initial_dist.event_shape

    @property
    def initial_dist(self):
        return JointDistribution(self.sub_process.initial_dist, super(AffineHierarchalProcess, self).initial_dist)

    def _inc_builder(self):
        return JointDistribution(self.sub_process.increment_dist(), self._increment_dist())

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> JointState:
        return JointState(
            sub_state=self.sub_process.initial_sample(shape),
            main_state=super(AffineHierarchalProcess, self).initial_sample(shape)
        )

    def mean_scale(self, x: JointState, parameters=None):
        sub_mean, sub_scale = self.sub_process.mean_scale(x["sub_state"])
        main_mean, main_scale = super(AffineHierarchalProcess, self).mean_scale(x)

        sub_unsqueeze = x["sub_state"].event_dim.numel() == 1
        main_unsqueeze = x["main_state"].event_dim.numel() == 1

        mean = (
            sub_mean.unsqueeze(-1) if sub_unsqueeze else sub_mean,
            main_mean.unsqueeze(-1) if main_unsqueeze else main_mean
        )

        scale = (
            sub_scale.unsqueeze(-1) if sub_unsqueeze else sub_scale,
            main_scale.unsqueeze(-1) if main_unsqueeze else main_scale
        )

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)
