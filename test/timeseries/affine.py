import pytest as pt
import torch
from stochproc import distributions as dists, timeseries as ts
from torch.distributions import Normal
from .stochastic_process import initial_distribution


class TestAffineTimeseries(object):
    samples = 100

    def test_affine_0d(self, initial_distribution):
        def mean_scale(x_, alpha, sigma):
            return alpha * x_.values, sigma

        params = [
            ts.NamedParameter("alpha", torch.tensor(1.0)),
            ts.NamedParameter("sigma", torch.tensor(0.05))
        ]

        increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        process = ts.AffineProcess(mean_scale, params, initial_distribution, increment_dist)

        x = process.initial_sample()

        for t in range(self.samples):
            x = process.propagate(x)

        path = process.sample_path(self.samples)
        assert path.shape == torch.Size([self.samples])
