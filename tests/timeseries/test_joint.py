import torch

from stochproc import timeseries as ts
from .test_affine import SAMPLES
import pytest

from .constants import BATCH_SHAPES
from .test_stochastic_process import assert_same_contents


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

        assert x.shape == torch.Size([SAMPLES]) + batch_shape + torch.Size([2])

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_affine_joint_processes(self, batch_shape, processes):
        joint_proc = ts.AffineJointStochasticProcess(**processes)
        x = joint_proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES]) + batch_shape + torch.Size([2])

    def test_parameters(self, processes):
        joint = ts.joint_process(**processes)

        parameters = joint.yield_parameters()

        for k, v in joint.sub_processes.items():
            for ps, pv in parameters.items():
                assert parameters[ps][k] is pv[k]

        overrides = {
            "mean": [2.0 * t for t in joint.sub_processes["mean"].parameters],
            "log_scale": [2.0 * t for t in joint.sub_processes["log_scale"].parameters]
        }

        # TODO: perhaps improve
        with joint.override_parameters(overrides):
            for k, v in overrides.items():
                assert_same_contents(joint.sub_processes[k].parameters, v)
        
        for k, v in overrides.items():
            assert_same_contents(joint.sub_processes[k].parameters, v, assert_is=False)
