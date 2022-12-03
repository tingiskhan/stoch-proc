import torch
import pyro.distributions as tdists
import pytest
from stochproc.torch import timeseries as ts, distributions as dists
from .test_affine import SAMPLES
from .constants import BATCH_SHAPES


class TestLinear(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_multidimensional(self, batch_shape):
        dim = torch.Size([5])
        initial_dist = increment_dist = dists.DistributionModule(tdists.Normal, loc=0.0, scale=1.0).expand(dim).to_event(1)

        a = torch.rand((initial_dist.shape[0], initial_dist.shape[0]))
        a /= a.sum(dim=-1).unsqueeze(0)

        b = a[0]
        a = a
        sigma = 0.05

        proc = ts.LinearModel(a, sigma, b=b, initial_dist=initial_dist, increment_dist=increment_dist)
        x = proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, *dim])
