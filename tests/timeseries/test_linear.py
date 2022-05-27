import torch
import pyro.distributions as tdists
import pytest
from stochproc import timeseries as ts, distributions as dists, NamedParameter
from .test_affine import SAMPLES
from .constants import BATCH_SHAPES


class TestLinear(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_multidimensional(self, batch_shape):
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

        x = proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, dim])
