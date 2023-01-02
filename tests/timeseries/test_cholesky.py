import pytest
import torch

from pyro.distributions import Normal
from stochproc import distributions as dists, timeseries as ts

from .constants import SAMPLES, BATCH_SHAPES


def mean_scale(x_, alpha, sigma):
    return alpha * x_.value, sigma * torch.eye(2, device=x_.value.device)


def initial_kernel(loc, scale):
    return Normal(loc, scale).expand(torch.Size([2])).to_event(1)


@pytest.fixture()
def process() -> ts.AffineProcess:
    params = [
        1.0,
        0.05
    ]

    increment_dist = Normal(loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)

    return ts.LowerCholeskyAffineProcess(mean_scale, params, increment_dist, initial_kernel, initial_parameters=(0.0, 1.0))


class TestAffineTimeseriesOneDimensional(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_cholesky(self, batch_shape, process):
        x = process.initial_sample(batch_shape)

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_states(SAMPLES, samples=batch_shape).get_path()
        assert path.shape == torch.Size([SAMPLES]) + batch_shape + process.event_shape

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_cholesky_joint(self, batch_shape, process):
        process = ts.joint_process(sub=ts.models.RandomWalk(0.05), main=process)

        x = process.initial_sample(batch_shape)

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_states(SAMPLES, samples=batch_shape).get_path()
        assert path.shape == torch.Size([SAMPLES]) + batch_shape + process.event_shape

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_cholesky_hierarchical(self, batch_shape, process):
        process = process.add_sub_process(ts.models.RandomWalk(0.05))

        x = process.initial_sample(batch_shape)

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_states(SAMPLES, samples=batch_shape).get_path()
        assert path.shape == torch.Size([SAMPLES]) + batch_shape + process.event_shape