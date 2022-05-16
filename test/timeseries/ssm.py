import pytest as pt
from stochproc import timeseries as ts, distributions as dists
import torch
from .affine import SAMPLES


class TestSSM(object):
    def test_ssm(self):
        rw = ts.models.RandomWalk(0.05)
        ssm = ts.LinearGaussianObservations(rw, torch.tensor([1.0, 0.01]))

        x, y = ssm.sample_path(SAMPLES)

        assert (x.shape[0] == y.shape[0] == SAMPLES)

    def test_joint_ssm(self):
        loc_1 = ts.models.RandomWalk(0.05)
        loc_2 = ts.models.RandomWalk(0.025)

        joint = ts.AffineJointStochasticProcess(loc_1=loc_1, loc_2=loc_2)

        ssm = ts.LinearGaussianObservations(joint, torch.eye(2))

        batch_size = torch.Size([10, 10, 2])
        x, y = ssm.sample_path(SAMPLES, samples=batch_size)

        assert (y.shape == torch.Size([SAMPLES, *batch_size, 2]))
