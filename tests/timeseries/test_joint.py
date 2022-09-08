import torch

from stochproc import timeseries as ts
from .test_affine import SAMPLES
import pytest

from .constants import BATCH_SHAPES


class TestJointProcesses(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_joint_processes(self, batch_shape):
        mean = ts.models.RandomWalk(0.05)
        log_scale = ts.models.OrnsteinUhlenbeck(0.025, 0.0, 0.05)

        joint_proc = ts.JointStochasticProcess(mean=mean, log_scale=log_scale)
        x = joint_proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, 2])

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_affine_joint_processes(self, batch_shape):
        mean = ts.models.RandomWalk(0.05)
        log_scale = ts.models.OrnsteinUhlenbeck(0.025, 0.0, 0.05)

        joint_proc = ts.AffineJointStochasticProcess(mean=mean, log_scale=log_scale)
        x = joint_proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, 2])

