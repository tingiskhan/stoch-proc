import torch

from stochproc import timeseries as ts
from .test_affine import SAMPLES


class TestJointProcesses(object):
    def test_joint_processes(self):
        mean = ts.models.RandomWalk(0.05)
        log_scale = ts.models.OrnsteinUhlenbeck(0.025, 0.0, 0.05)

        joint_proc = ts.AffineJointStochasticProcess(mean=mean, log_scale=log_scale)
        x = joint_proc.sample_path(SAMPLES)

        assert x.shape == torch.Size([SAMPLES, 2])

