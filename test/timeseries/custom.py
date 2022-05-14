import pytest as pt
from stochproc.timeseries import models as mods
from .affine import SAMPLES


def models():
    yield mods.OrnsteinUhlenbeck(0.025, 0.0, 0.05)
    yield mods.Verhulst(0.025, 1.0, 0.05, 0.1, num_steps=10)


class TestCustomModels(object):
    def test_all_models(self):
        for proc in models():
            x = proc.sample_path(SAMPLES)

            assert x.shape[0] == SAMPLES
