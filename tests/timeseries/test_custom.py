import itertools

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


class TestCustomModels(object):
    @pytest.mark.parametrize("batch_size, model", tuple(itertools.product(BATCH_SHAPES, models())))
    def test_all_models(self, batch_size, model):
        x = model.sample_states(SAMPLES, samples=batch_size).get_path()

        assert (x.shape == torch.Size([SAMPLES, *batch_size, *model.event_shape])) and ~torch.isnan(x).any()

