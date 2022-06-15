import itertools

import pyro
import pytest
import torch

from stochproc.timeseries import models, StateSpaceModel
from pyro.distributions import LogNormal, Normal


def do_infer_with_pyro(model, data, num_samples=1_000):
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optim = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())

    niter = 10_000
    pyro.clear_param_store()

    for n in range(niter):
        loss = svi.step(data)

    posterior_predictive = pyro.infer.Predictive(
        model,
        guide=guide,
        num_samples=num_samples
    )

    return guide, posterior_predictive


class TestPyroIntegration(object):
    @pytest.mark.parametrize("mode", ["full", "approximate"])
    def test_sample_model(self, mode):
        sigma = pyro.sample("sigma", LogNormal(loc=-2.0, scale=0.5))
        model = models.RandomWalk(sigma)

        length = 100
        latent = model.do_sample_pyro(pyro, length, mode=mode)

        assert latent.shape == torch.Size([length])

    def test_infer_parameters_only(self):
        pyro.set_rng_seed(123)
        true_sigma = 0.05

        linear_model = models.RandomWalk(true_sigma)
        x = linear_model.sample_states(100).get_path()

        def pyro_model(data):
            sigma = pyro.sample("sigma", LogNormal(loc=-2.0, scale=0.5))
            model_ = models.RandomWalk(sigma)
            model_.do_sample_pyro(pyro, obs=data)

        guid, posterior_predictive = do_infer_with_pyro(pyro_model, x)

        posterior_draws = posterior_predictive(x)

        mean = posterior_draws["sigma"].mean()
        std = posterior_draws["sigma"].std()

        assert (mean - 2.0 * std) <= true_sigma <= (mean + 2.0 * std)

    @pytest.mark.parametrize("mode, observe_every_step", itertools.product(["full", "approximate"], [1, 5]))
    def test_sample_ssm(self, mode, observe_every_step):
        sigma = 0.05
        model = models.RandomWalk(sigma)

        def build_obs(x):
            return Normal(loc=x.values, scale=0.1)

        ssm = StateSpaceModel(model, build_obs, (), observe_every_step=observe_every_step)

        length = 100
        latent = ssm.do_sample_pyro(pyro, length, mode=mode)

        assert latent.shape == torch.Size([length + 1])
