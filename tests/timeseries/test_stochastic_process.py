import pytest as pt
import torch
from torch.distributions import Normal

from stochproc import timeseries as ts, distributions as dists


def initial_distribution(*args) -> dists.DistributionModule:
    return Normal(loc=0.0, scale=1.0)


class TestStochasticProcess(object):
    def test_same_parameter_same_module(self):
        val = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        parameters = [
            val,
            val
        ]

        sts = ts.StructuralStochasticProcess(None, parameters, initial_distribution)
        val += 1.0

        assert (sts.parameters[0] is val) and (sts.parameters[1] is sts.parameters[0])

        val.data = torch.empty(2000).normal_()

        assert (sts.parameters[0] is val) and (sts.parameters[1] is sts.parameters[0])
        assert (sts.parameters[0] is val) and (sts.parameters[0].shape == val.shape)

    def test_same_parameter_different_process(self):
        val = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

        parameters = [
            val,
        ]

        sts_1 = ts.StructuralStochasticProcess(None, parameters, initial_distribution)
        sts_2 = ts.StructuralStochasticProcess(None, parameters, initial_distribution)

        assert (sts_1 is not sts_2) and (sts_2.parameters[0] is sts_1.parameters[0])
        assert len(tuple(sts_1.parameters)) == len(tuple(sts_2.parameters)) == 1
