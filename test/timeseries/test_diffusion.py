import math

import torch
import torch.distributions as tdists

from stochproc import distributions as dists, timeseries as ts, NamedParameter
from .test_affine import SAMPLES


class TestDiffusionOneDimensional(object):
    def test_affine_euler(self):
        dt = 0.05

        def dynamics(x_, kappa, gamma, sigma):
            return kappa * (gamma - x_.values), sigma

        parameters = (
            NamedParameter("kappa", 0.05),
            NamedParameter("gamma", 0.0),
            NamedParameter("sigma", 0.075)
        )

        def builder(kappa, gamma, sigma):
            return tdists.Normal(loc=gamma, scale=sigma / (2 * kappa).sqrt())

        initial_dist = dists.DistributionModule(builder, kappa=parameters[0], gamma=parameters[1], sigma=parameters[2])
        increment_dist = dists.DistributionModule(tdists.Normal, loc=0.0, scale=math.sqrt(dt))

        discretized_ou = ts.AffineEulerMaruyama(
            dynamics,
            parameters,
            dt=dt,
            initial_dist=initial_dist,
            increment_dist=increment_dist,
            num_steps=20
        )

        x = discretized_ou.sample_path(SAMPLES)

        assert (x.shape == torch.Size([SAMPLES])) and (not torch.isnan(x).any())

    def test_runge_kutta(self):
        dt = 0.05

        def dynamics(x_, kappa, gamma):
            return kappa * (gamma - x_.values)

        parameters = (
            NamedParameter("kappa", 0.05),
            NamedParameter("gamma", 0.0),
        )

        for tuning_std in [False, 0.1]:
            newton_cooling = ts.RungeKutta(
                dynamics, parameters, initial_values=torch.tensor(1.0), dt=dt, event_dim=0, tuning_std=tuning_std
            )
            x = newton_cooling.sample_path(SAMPLES)

            assert x.shape == torch.Size([SAMPLES])