import torch

from . import AffineProcess, AffineJointStochasticProcess


class AffineHierarchalProcess(AffineJointStochasticProcess):
    r"""
    Defines a "hierarchal" affine process, by which we mean a process that ... TODO

    """

    def __init__(self, sub_process: AffineProcess, main_process: AffineProcess):
        """
        Initializes the :class:`AffineHierarchalProcess` class.

        Args:
            sub_process: child/sub process.
            main_process: main process.
        """

        super(AffineHierarchalProcess, self).__init__(sub=sub_process, main=main_process)

    def mean_scale(self, x, parameters=None):
        sub_mean, sub_scale = self.sub_processes["sub"].mean_scale(x["sub"])
        main_mean, main_scale = self.sub_processes["main"].mean_scale(x)

        sub_unsqueeze = x["sub"].event_dim.numel() == 1
        main_unsqueeze = x["main"].event_dim.numel() == 1

        mean = (
            sub_mean.unsqueeze(-1) if sub_unsqueeze else sub_mean,
            main_mean.unsqueeze(-1) if main_unsqueeze else main_mean
        )

        scale = (
            sub_scale.unsqueeze(-1) if sub_unsqueeze else sub_scale,
            main_scale.unsqueeze(-1) if main_unsqueeze else main_scale
        )

        return torch.cat(mean, dim=-1), torch.cat(scale, dim=-1)
