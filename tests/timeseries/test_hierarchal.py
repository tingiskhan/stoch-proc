import torch
from torch.distributions import Normal

from stochproc import timeseries as ts, distributions as dists
from .test_affine import SAMPLES
import pytest

from .constants import BATCH_SHAPES


def mean_scale(x, beta, sigma):
    return x["sub_state"].values + beta * x["main_state"].values, sigma


class TestHierarchalProcess(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_hierarchal_processes(self, batch_shape):
        sub = ts.models.AR(0.0, 0.99, 0.001)

        inc_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
        main = ts.AffineHierarchalProcess(sub, mean_scale, (0.99, 0.05), initial_dist=inc_dist, increment_dist=inc_dist)

        assert main.event_shape == torch.Size([2])

        x = main.sample_states(SAMPLES, batch_shape).get_path()

        assert x.shape == torch.Size([SAMPLES, *batch_shape, *main.event_shape])

