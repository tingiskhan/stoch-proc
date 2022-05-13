import pytest as pt
from stochproc import distributions as dists, timeseries as ts, NamedParameter
import torch.distributions as tdists
import torch
import math
from .affine import SAMPLES


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

        initial_dist = dists.DistributionModule(tdists.Normal, loc=0.0, scale=1.0)
        increment_dist = dists.DistributionModule(tdists.Normal, loc=0.0, scale=math.sqrt(dt))

        def initial_transform(module: ts.AffineEulerMaruyama, base_dist):
            kappa, gamma, sigma = module.functional_parameters()
            return tdists.TransformedDistribution(base_dist, tdists.AffineTransform(gamma, sigma / (2 * kappa).sqrt()))

        discretized_ou = ts.AffineEulerMaruyama(
            dynamics,
            parameters,
            dt=dt,
            initial_dist=initial_dist,
            increment_dist=increment_dist,
            initial_transform=initial_transform,
            num_steps=20
        )

        x = discretized_ou.sample_path(SAMPLES)

        assert (x.shape == torch.Size([SAMPLES])) and (not torch.isnan(x).any())
