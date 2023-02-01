import itertools
from math import pi

import pytest
import torch

from stochproc.timeseries import models as mods
from .test_affine import SAMPLES
from .constants import BATCH_SHAPES


def models():
    yield mods.OrnsteinUhlenbeck(0.025, 0.0, 0.05)
    yield mods.Verhulst(0.025, 1.0, 0.05, 0.1)
    yield mods.RandomWalk(0.05)
    yield mods.LocalLinearTrend(torch.tensor([0.01, 0.05]))
    yield mods.AR(0.0, 1.0, 0.05)
    yield mods.AR(0.0, torch.tensor([0.25, 0.05, 0.01, 0.01, -0.02]), 0.05, lags=5)
    yield mods.UCSV(0.025)
    yield mods.Seasonal(12, 0.05)
    yield mods.SmoothLinearTrend(mods.RandomWalk(0.05))
    yield mods.SmoothLinearTrend(mods.OrnsteinUhlenbeck(0.025, 0.0, 0.05))
    yield mods.TrendingOU(0.01, 0.03, 0.2, 1.0)
    yield mods.SelfExcitingLatentProcesses(0.01, 2.0, 0.05, 0.1, 3.0, 2.0, dt=0.05)
    yield mods.HarmonicProcess(3, 0.05)
    yield mods.CyclicalProcess(0.98, 2.0 * pi / 1_000, 0.25)


class TestCustomModels(object):
    @pytest.mark.parametrize("batch_size", BATCH_SHAPES)
    @pytest.mark.parametrize("model", models())
    def test_all_models(self, batch_size, model):
        states = model.sample_states(SAMPLES, samples=batch_size)
        x = states.get_path()

        assert all(s.batch_shape == batch_size for s in states)
        assert (x.shape == torch.Size([SAMPLES]) + batch_size + model.event_shape) and ~torch.isnan(x).any()
