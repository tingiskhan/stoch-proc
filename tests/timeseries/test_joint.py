import torch

from stochproc import timeseries as ts
from .test_affine import SAMPLES
import pytest

from .constants import BATCH_SHAPES


class CustomDict(dict):
    keys_accessed = set()

    def __getitem__(self, __key):
        self.keys_accessed.add(__key)
        return super().__getitem__(__key)


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

        # TODO: Perhaps use mock here...
        overrides = CustomDict(
            mean=[2.0 * t for t in joint.sub_processes["mean"].parameters],
            log_scale=[2.0 * t for t in joint.sub_processes["log_scale"].parameters]
        )

        x = joint.initial_sample()
        x_new = joint.build_density(x, overrides)

        assert set(joint.sub_processes.keys()) == set(overrides.keys_accessed)
