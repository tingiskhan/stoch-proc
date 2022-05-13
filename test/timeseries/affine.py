import pytest as pt
import torch
from stochproc import distributions as dists, timeseries as ts, NamedParameter
from torch.distributions import Normal
from .stochastic_process import initial_distribution


class TestAffineTimeseries0Dimension(object):
    samples = 100

    @classmethod
    def mean_scale(cls, x_, alpha, sigma):
        return alpha * x_.values, sigma

    def test_affine_0d(self, initial_distribution):
        params = [
            NamedParameter("alpha", 1.0),
            NamedParameter("sigma", 0.05)
        ]

        increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        process = ts.AffineProcess(self.mean_scale, params, initial_distribution, increment_dist)

        x = process.initial_sample()

        for t in range(self.samples):
            x = process.propagate(x)

        path = process.sample_path(self.samples)
        assert path.shape == torch.Size([self.samples])

    def test_affine_0d_batched(self, initial_distribution):
        params = [
            NamedParameter("alpha", dists.Prior(Normal, loc=0.0, scale=1.0)),
            NamedParameter("sigma", 0.05)
        ]

        increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        process = ts.AffineProcess(self.mean_scale, params, initial_distribution, increment_dist)

        size = torch.Size([1_000, 10, 20])
        process.sample_params_(size)

        path = process.sample_path(self.samples)

        assert path.shape == torch.Size([self.samples, *size])
