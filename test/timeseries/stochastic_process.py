from stochproc import timeseries as ts, distributions as dists, NamedParameter
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
            NamedParameter("alpha", val)
        ]

        sts = ts.StructuralStochasticProcess(parameters, initial_distribution)

        assert "alpha" in sts.buffer_dict
        assert sts.buffer_dict["alpha"] is val

        assert (sts.n_dim == 0) and (sts.num_vars == 1)

        initial_sample = sts.initial_sample()
        assert callable(initial_sample._values) and isinstance(initial_sample.values, torch.Tensor)

    def test_initialize_sts_prior(self, initial_distribution: dists.DistributionModule):
        val = dists.Prior(Normal, loc=0.0, scale=1.0)

        parameters = [
            NamedParameter("alpha", val),
            NamedParameter("beta", val)
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

    def test_same_prior_same_parameter(self):
        loc_parameter = NamedParameter("loc", dists.Prior(Normal, loc=0.0, scale=1.0))
        scale_parameter = NamedParameter("scale", 0.05)

        dist = dists.DistributionModule(Normal, loc=loc_parameter, scale=scale_parameter)

        proc = ts.StructuralStochasticProcess((loc_parameter, scale_parameter), initial_dist=dist)

        assert (
                (proc.parameter_dict["loc"] is dist.parameter_dict["loc"]) and
                (len(tuple(proc.parameters())) == 1) and
                (proc.buffer_dict["scale"] is dist.buffer_dict["scale"])
        )

    def test_incongruent_names(self):
        loc_parameter = NamedParameter("alpha", dists.Prior(Normal, loc=0.0, scale=1.0))
        scale_parameter = NamedParameter("scale", 0.05)

        with pt.raises(Exception):
            dist = dists.DistributionModule(Normal, loc=loc_parameter, scale=scale_parameter)

    def test_nested_parameters(self):
        loc_parameter = NamedParameter("loc", dists.Prior(Normal, loc=0.0, scale=1.0))
        scale_parameter = NamedParameter("scale", dists.Prior(Normal, loc=0.0, scale=1.0))

        dist = dists.DistributionModule(Normal, loc=loc_parameter, scale=1.0)
        proc = ts.StructuralStochasticProcess((scale_parameter,), initial_dist=dist)

        assert (
            len(tuple(proc.parameters_and_priors())) == 2
        )
