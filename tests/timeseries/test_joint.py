import torch

from stochproc import timeseries as ts
from .test_affine import SAMPLES
import pytest

from .constants import BATCH_SHAPES


@pytest.fixture()
def processes():
    mean = ts.models.RandomWalk(0.05)
    log_scale = ts.models.OrnsteinUhlenbeck(0.025, 0.0, 0.05)

    return {
        "mean": mean,
        "log_scale": log_scale
    }


class TestJointProcesses(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_joint_processes(self, batch_shape, processes):
        joint_proc = ts.JointStochasticProcess(**processes)
        x = joint_proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, 2])

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_affine_joint_processes(self, batch_shape, processes):
        joint_proc = ts.AffineJointStochasticProcess(**processes)
        x = joint_proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, 2])

    def test_parameters(self, processes):
        joint = ts.joint_process(**processes)

        parameters = tuple(joint.yield_parameters())

        sub_process_parameters = sum((p.parameters for p in processes.values()), ())
        sub_process_initial_parameters = sum((p.initial_parameters for p in processes.values()), ())
        assert len(parameters) == len(set(sub_process_parameters + sub_process_initial_parameters))
