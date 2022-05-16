import pytest as pt
from stochproc.timeseries import models as mods
import torch
from .affine import SAMPLES


def models():
    yield mods.OrnsteinUhlenbeck(0.025, 0.0, 0.05)
    yield mods.Verhulst(0.025, 1.0, 0.05, 0.1, num_steps=10)
    yield mods.RandomWalk(0.05)
    yield mods.LocalLinearTrend(torch.tensor([0.01, 0.05]))
    yield mods.AR(0.0, 1.0, 0.05)
    yield mods.AR(0.0, torch.tensor([0.25, 0.05, 0.01, 0.01, -0.02]), 0.05, lags=5)
    yield mods.UCSV(0.025)


class TestCustomModels(object):
    def test_all_models(self):
        for proc in models():
            x = proc.sample_path(SAMPLES)

            assert (x.shape[0] == SAMPLES) and ~torch.isnan(x).any()

    def test_all_models_batched(self):
        batch_size = torch.Size([15, 10])
        for proc in models():
            x = proc.sample_path(SAMPLES, batch_size)

            assert (x.shape[:3] == torch.Size([SAMPLES, *batch_size]))
