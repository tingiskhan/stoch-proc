import pytest as pt
from stochproc import timeseries as ts, distributions as dists
import torch
from .affine import SAMPLES


class TestSSM(object):
    def test_ssm(self):
        rw = ts.models.RandomWalk(torch.tensor([0.05, 0.01]), initial_mean=torch.zeros(2))
        ssm = ts.LinearGaussianObservations(rw, torch.tensor([1.0, 0.01]))

        x, y = ssm.sample_path(SAMPLES)

        assert (x.shape[0] == y.shape[0] == SAMPLES)
