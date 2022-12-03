import pytest
import torch
from pyro.distributions import Normal

from stochproc.torch import distributions as dists, timeseries as ts
from .test_stochastic_process import initial_distribution   # flake8: noqa
from .constants import SAMPLES, BATCH_SHAPES


def mean_scale(x_, alpha, sigma):
    return alpha * x_.values, sigma


class TestAffineTimeseriesOneDimensional(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_affine_0d(self, initial_distribution, batch_shape):
        params = [
            1.0,
            0.05
        ]

        increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        process = ts.AffineProcess(mean_scale, params, initial_distribution, increment_dist)

        x = process.initial_sample(batch_shape)

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_states(SAMPLES, samples=batch_shape).get_path()
        assert path.shape == torch.Size([SAMPLES, *batch_shape])


class TestAffineTimeseriesMultiDimensional(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_affine_2d(self, batch_shape):
        params = [
            torch.ones(2),
            0.05
        ]

        increment_dist = initial_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)

        process = ts.AffineProcess(mean_scale, params, initial_dist, increment_dist)
        x = process.initial_sample(batch_shape)

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_states(SAMPLES, samples=batch_shape).get_path()
        assert path.shape == torch.Size([SAMPLES, *batch_shape, 2])
