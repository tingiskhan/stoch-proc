from stochproc import timeseries as ts, distributions as dists
import pytest as pt
import torch
from torch.distributions import Normal


@pt.fixture()
def initial_distribution() -> dists.DistributionModule:
    return dists.DistributionModule(Normal, loc=0.0, scale=1.0)


class TestStochasticProcess(object):
    def test_initialize_sts_fix_parameter(self, initial_distribution: dists.DistributionModule):
        val = torch.tensor(1.0)

        parameters = [
            ts.NamedParameter("alpha", val)
        ]

        sts = ts.StructuralStochasticProcess(parameters, initial_distribution)

        assert "alpha" in sts.buffer_dict
        assert sts.buffer_dict["alpha"] is val

        assert (sts.n_dim == 0) and (sts.num_vars == 1)

        initial_sample = sts.initial_sample()
        assert isinstance(initial_sample.distribution, Normal)

    def test_initialize_sts_prior(self, initial_distribution: dists.DistributionModule):
        val = dists.Prior(Normal, loc=0.0, scale=1.0)

        parameters = [
            ts.NamedParameter("alpha", val),
            ts.NamedParameter("beta", val)
        ]

        sts = ts.StructuralStochasticProcess(parameters, initial_distribution)

        assert ("alpha" in sts.parameter_dict) and ("alpha" not in sts.buffer_dict)
        assert (
                ("alpha" in sts.prior_dict) and
                ("beta" in sts.prior_dict) and
                (sts.prior_dict["alpha"] is sts.prior_dict["beta"])
        )

        size = torch.Size([1_000, 10])
        sts.sample_params_(size)

        assert (sts.parameter_dict["alpha"].shape == size)
