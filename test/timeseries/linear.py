from stochproc import timeseries as ts, distributions as dists, NamedParameter
import pytest as pt
import torch.distributions as tdists
import torch
from .affine import SAMPLES


class TestLinear(object):
    def test_multidimensional(self):
        dim = 5
        initial_dist = increment_dist = dists.DistributionModule(
            tdists.Normal, loc=torch.zeros(dim), scale=torch.ones(dim), reinterpreted_batch_ndims=1
        )

        a = torch.rand((initial_dist.shape[0], initial_dist.shape[0]))
        a /= a.sum(dim=-1).unsqueeze(0)

        b = NamedParameter("b", a[0])
        a = NamedParameter("a", a)
        sigma = NamedParameter("scale", 0.05)

        proc = ts.LinearModel(a, sigma, b=b, initial_dist=initial_dist, increment_dist=increment_dist)

        batch_size = torch.Size([10, 15])
        x = proc.sample_path(SAMPLES, samples=batch_size)

        assert x.shape == torch.Size([SAMPLES, *batch_size, dim])

