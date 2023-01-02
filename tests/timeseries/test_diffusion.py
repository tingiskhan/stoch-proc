import math
import pytest

import torch
import torch.distributions as tdists

from stochproc import distributions as dists, timeseries as ts
from .constants import SAMPLES, BATCH_SHAPES


class TestDiffusionOneDimensional(object):
    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_affine_euler(self, batch_shape):
        dt = 0.05

        def dynamics(x_, kappa, gamma, sigma):
            return kappa * (gamma - x_.value), sigma

        parameters = (
            0.05,
            0.0,
            0.075
        )

        def initial_kernel(kappa, gamma, sigma):
            return tdists.Normal(loc=gamma, scale=sigma / (2 * kappa).sqrt())

        increment_dist = tdists.Normal(loc=0.0, scale=math.sqrt(dt))
        discretized_ou = ts.AffineEulerMaruyama(dynamics, parameters, increment_dist, dt, initial_kernel)

        x = discretized_ou.sample_states(SAMPLES, samples=batch_shape).get_path()

        assert (x.shape == torch.Size([SAMPLES]) + batch_shape) and (not torch.isnan(x).any())

    @pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
    def test_runge_kutta(self, batch_shape):
        dt = 0.05

        def dynamics(x_, kappa, gamma):
            return kappa * (gamma - x_.value)

        parameters = (
            0.05,
            0.0,
        )

        for tuning_std in [False, 0.1]:
            newton_cooling = ts.RungeKutta(
                dynamics, parameters, initial_values=torch.tensor(1.0), dt=dt, event_dim=0, tuning_std=tuning_std
            )
            x = newton_cooling.sample_states(SAMPLES, samples=batch_shape).get_path()

            assert x.shape == torch.Size([SAMPLES]) + batch_shape
