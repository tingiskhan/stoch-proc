import torch
from torch.distributions import Normal

from stochproc import timeseries as ts, distributions as dists
from .test_affine import SAMPLES
import pytest

from math import sqrt
from pyro.distributions import Normal, LogNormal

from .constants import BATCH_SHAPES


def mean_scale(x, beta, sigma):
    return x["sub"].values + beta * x.values, sigma


class TestHierarchalProcess(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_hierarchal_processes(self, batch_shape):
        sub = ts.models.AR(0.0, 0.99, 0.001)

        inc_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        main = ts.AffineProcess(mean_scale, (0.99, 0.05), initial_dist=inc_dist, increment_dist=inc_dist)

        hieararchal = ts.AffineHierarchalProcess(sub, main)

        assert hieararchal.event_shape == torch.Size([2])

        x = hieararchal.sample_states(SAMPLES, batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, *hieararchal.event_shape])

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_hull_white(self, batch_shape):
        def mean_scale(x, kappa, theta, sigma):
            return kappa * (theta - x["sub"].values / kappa - x.values), sigma

        dt = 1.0
        u = ts.models.OrnsteinUhlenbeck(0.01, 0.0, 0.01, dt=dt)

        inc_dist = dists.DistributionModule(Normal, loc=0.0, scale=sqrt(dt))
        init_dist = dists.DistributionModule(LogNormal, loc=-2.0, scale=0.5)
        hull_white = ts.AffineEulerMaruyama(mean_scale, (0.01, 0.5, 0.05), init_dist, inc_dist, dt).add_sub_process(u)

        x = hull_white.sample_states(5 * SAMPLES, batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, *hull_white.event_shape])