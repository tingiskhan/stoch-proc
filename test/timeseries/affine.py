import pytest as pt
import torch
from stochproc import distributions as dists, timeseries as ts, NamedParameter
from torch.distributions import Normal
from .stochastic_process import initial_distribution

SAMPLES = 100


def mean_scale(x_, alpha, sigma):
    return alpha * x_.values, sigma


class TestAffineTimeseriesOneDimensional(object):
    def test_affine_0d(self, initial_distribution):
        params = [
            NamedParameter("alpha", 1.0),
            NamedParameter("sigma", 0.05)
        ]

        increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        process = ts.AffineProcess(mean_scale, params, initial_distribution, increment_dist)

        x = process.initial_sample()

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_path(SAMPLES)
        assert path.shape == torch.Size([SAMPLES])

    def test_affine_0d_batched(self, initial_distribution):
        params = [
            NamedParameter("alpha", dists.Prior(Normal, loc=0.0, scale=1.0)),
            NamedParameter("sigma", 0.05)
        ]

        increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        process = ts.AffineProcess(mean_scale, params, initial_distribution, increment_dist)

        size = torch.Size([1_000, 10, 20])
        process.sample_params_(size)

        path = process.sample_path(SAMPLES)

        assert path.shape == torch.Size([SAMPLES, *size])


class TestAffineTimeseriesMultiDimensional(object):
    def test_affine_2d(self):
        params = [
            NamedParameter("alpha", torch.ones(2)),
            NamedParameter("sigma", 0.05)
        ]

        increment_dist = initial_dist = dists.DistributionModule(
            Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1
        )

        process = ts.AffineProcess(mean_scale, params, initial_dist, increment_dist)
        x = process.initial_sample()

        for t in range(SAMPLES):
            x = process.propagate(x)

        path = process.sample_path(SAMPLES)
        assert path.shape == torch.Size([SAMPLES, 2])

    def test_affine_2d_batched(self):
        params = [
            NamedParameter("alpha", dists.Prior(Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1)),
            NamedParameter("sigma", 0.05)
        ]

        increment_dist = initial_dist = dists.DistributionModule(
            Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1
        )

        process = ts.AffineProcess(mean_scale, params, initial_dist, increment_dist)

        size = torch.Size([1_000, 10, 20])
        process.sample_params_(size)

        path = process.sample_path(SAMPLES)

        assert path.shape == torch.Size([SAMPLES, *size, 2])
