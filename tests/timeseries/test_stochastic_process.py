import pytest as pt
import torch
from torch.distributions import Normal

from stochproc import timeseries as ts, distributions as dists


@pt.fixture()
def initial_distribution() -> dists.DistributionModule:
    return dists.DistributionModule(Normal, loc=0.0, scale=1.0)


class TestStochasticProcess(object):
    def test_initialize_sts_fix_parameter(self, initial_distribution: dists.DistributionModule):
        val = torch.tensor(1.0)

        parameters = [
            val
        ]

        sts = ts.StructuralStochasticProcess(parameters, initial_distribution)

        assert "0" in sts.buffer_dict
        assert sts.buffer_dict["0"] is val

        assert (sts.n_dim == 0) and (sts.num_vars == 1)

        initial_sample = sts.initial_sample()
        assert callable(initial_sample._values) and isinstance(initial_sample.value, torch.Tensor)

    def test_same_parameter_same_module(self, initial_distribution):
        val = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        parameters = [
            val,
            val
        ]

        sts = ts.StructuralStochasticProcess(parameters, initial_distribution)

        val += 1.0

        assert ("0" in sts.parameter_dict) and ("1" in sts.parameter_dict)
        assert (sts.parameter_dict["0"] is val) and (sts.parameter_dict["1"] is sts.parameter_dict["0"])

        val.data = torch.empty(2000).normal_()

        assert (sts.parameter_dict["0"] is val) and (sts.parameter_dict["1"] is sts.parameter_dict["0"])
        assert (sts.parameter_dict["0"] is val) and (sts.parameter_dict["0"].shape == val.shape)

        for true_p, pointer in zip(sts.parameters(), sts.functional_parameters()):
            assert pointer is true_p

    def test_same_parameter_different_module(self, initial_distribution):
        val = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        parameters = [
            val,
        ]

        sts_1 = ts.StructuralStochasticProcess(parameters, initial_distribution)
        sts_2 = ts.StructuralStochasticProcess(parameters, initial_distribution)

        assert (sts_1 is not sts_2) and (sts_2.parameter_dict["0"] is sts_1.parameter_dict["0"])
        assert len(tuple(sts_1.parameter_dict)) == len(tuple(sts_2.parameter_dict)) == 1

        class CombinedModule(torch.nn.Module):
            def __init__(self, a, b):
                super().__init__()
                self.a = a
                self.b = b

        combined = CombinedModule(sts_1, sts_2)
        assert len(tuple(combined.parameters())) == 1

    def test_move_to_cuda(self, initial_distribution):
        if not torch.cuda.is_available():
            return

        val = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        parameters = [
            val,
            val ** 2.0,
        ]

        sts = ts.StructuralStochasticProcess(parameters, initial_distribution).cuda()

        for p1, p2 in zip(sts.parameters(), sts.functional_parameters()):
            assert p1 is p2
