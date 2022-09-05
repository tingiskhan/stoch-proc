import pytest
import torch

from pyro.distributions import Normal
from stochproc import distributions as dists, timeseries as ts

from .constants import SAMPLES, BATCH_SHAPES


def mean_scale(x_, alpha, sigma):
    return alpha * x_.values, sigma * torch.eye(2, device=x_.values.device)


class TestAffineTimeseriesOneDimensional(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_mv_affine(self, batch_shape):
        params = [
            1.0,
            0.05
        ]

        increment_dist = initial_distribution = dists.DistributionModule(Normal, loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)
        process = ts.LowerCholeskyAffineProcess(mean_scale, params, initial_distribution, increment_dist)

        x = process.initial_sample(batch_shape)

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_states(SAMPLES, samples=batch_shape).get_path()
        assert path.shape == torch.Size([SAMPLES, *batch_shape, *process.event_shape])