import pyro
import pytest
import torch

from stochproc import NamedParameter
from stochproc.timeseries import models, StateSpaceModel
from pyro.distributions import LogNormal, Normal


def do_infer_with_pyro(model, data):
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optim = pyro.optim.Adam({"lr": 0.01})
    svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())

    niter = 10_000
    pyro.clear_param_store()

    for n in range(niter):
        loss = svi.step(data)

    num_samples = 1_000
    posterior_predictive = pyro.infer.Predictive(
        model,
        guide=guide,
        num_samples=num_samples
    )

    return guide, posterior_predictive


@pytest.fixture
def linear_model():
    true_sigma = NamedParameter("scale", 0.05)
    return models.RandomWalk(true_sigma)


class TestPyroIntegration(object):
    @pytest.mark.parametrize("mode", ["full", "approximate"])
    def test_sample_model(self, mode):
        sigma = pyro.sample("sigma", LogNormal(loc=-2.0, scale=0.5))
        model = models.RandomWalk(sigma)

        length = 100
        latent = model.do_sample_pyro(pyro, length, mode=mode)

        assert latent.shape == torch.Size([length])

    def test_infer_parameters_only(self, linear_model):
        x = linear_model.sample_states(100).get_path()

        def pyro_model(data):
            sigma = pyro.sample("sigma", LogNormal(loc=-2.0, scale=0.5))
            model_ = models.RandomWalk(sigma)
            model_.do_sample_pyro(pyro, obs=data)

        guid, posterior_predictive = do_infer_with_pyro(pyro_model, x)

        posterior_draws = posterior_predictive(x)

        mean = posterior_draws["sigma"].mean()
        std = posterior_draws["sigma"].std()

        assert (mean - std) <= linear_model.parameters_and_buffers()["scale"] <= (mean + std)
