import itertools

import torch

from stochproc import timeseries as ts
from pyro.distributions import Normal
import pytest

from .test_affine import SAMPLES
from .constants import BATCH_SHAPES


SAMPLE_INITIAL = [True, False]
SAMPLE_EVERY = [1, 5]


class TestSSM(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    @pytest.mark.parametrize("sample_initial", SAMPLE_INITIAL)
    @pytest.mark.parametrize("sample_every", SAMPLE_EVERY)
    def test_ssm(self, batch_shape, sample_initial, sample_every):
        rw = ts.models.RandomWalk(0.05)

        def f(x_, a):
            return Normal(loc=x_.value.unsqueeze(-1) * a, scale=1.0).to_event(1)

        ssm = ts.StateSpaceModel(rw, f, parameters=(torch.tensor([1.0, 0.01]),), observe_every_step=sample_every)

        x_0 = rw.initial_sample(batch_shape) if sample_initial else None

        states = ssm.sample_states(SAMPLES, samples=batch_shape, x_0=x_0)
        x, y = states.get_paths()

        assert x.shape == torch.Size([SAMPLES]) + batch_shape
        assert y.shape == torch.Size([SAMPLES]) + batch_shape + torch.Size([2])

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_joint_ssm(self, batch_shape):
        loc_1 = ts.models.RandomWalk(0.05)
        loc_2 = ts.models.RandomWalk(0.025)

        joint = ts.joint_process(loc_1=loc_1, loc_2=loc_2)

        def f(x_, a):
            return Normal(loc=a.matmul(x_.value.unsqueeze(-1)).squeeze(-1), scale=1.0).to_event(1)

        ssm = ts.StateSpaceModel(joint, f, parameters=(torch.eye(2),))
        x, y = ssm.sample_states(SAMPLES, samples=batch_shape).get_paths()

        assert y.shape == torch.Size([SAMPLES]) + batch_shape + torch.Size([2])

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    @pytest.mark.parametrize("sample_initial", SAMPLE_INITIAL)
    @pytest.mark.parametrize("sample_every", SAMPLE_EVERY)
    def test_linear_ssm(self, batch_shape, sample_initial, sample_every):
        rw = ts.models.RandomWalk(0.05)
        ssm = ts.LinearStateSpaceModel(rw, (torch.tensor([1.0, 0.01]).unsqueeze(-1), 0.0, 1.0), torch.Size([2]), observe_every_step=sample_every)

        x_0 = rw.initial_sample(batch_shape) if sample_initial else None

        states = ssm.sample_states(SAMPLES, samples=batch_shape, x_0=x_0)
        x, y = states.get_paths()

        assert x.shape == torch.Size([SAMPLES]) + batch_shape
        assert y.shape == torch.Size([SAMPLES]) + batch_shape + torch.Size([2])
    