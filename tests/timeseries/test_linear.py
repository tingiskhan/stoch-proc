from functools import partial
import torch
import pyro.distributions as tdists
import pytest
from stochproc import timeseries as ts, distributions as dists
from .test_affine import SAMPLES
from .constants import BATCH_SHAPES


def initial_kernel(a, sigma, b, dim):
    return tdists.Normal(loc=0.0, scale=1.0).expand(dim).to_event(1)


class TestLinear(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_multidimensional(self, batch_shape):
        dim = torch.Size([5])

        init_kernel = partial(initial_kernel, dim=dim)
        increment_dist = tdists.Normal(loc=0.0, scale=1.0).expand(dim).to_event(1)

        a = torch.rand((dim[0], dim[0]))
        a /= a.sum(dim=-1).unsqueeze(0)

        b = a[0]
        a = a
        sigma = 0.05

        proc = ts.LinearModel((a, b, sigma), increment_dist, init_kernel)
        x = proc.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, *dim])
