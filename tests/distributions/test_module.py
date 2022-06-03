import torch

from stochproc.distributions import DistributionModule
from pyro.distributions import Normal


class TestDistributionModule(object):
    def test_move_to_cuda(self):
        if not torch.cuda.is_available():
            return

        mod = DistributionModule(Normal, loc=0.0, scale=1.0)

        for p in mod.buffers():
            assert p.device == torch.device("cpu")

        mod = mod.cuda()

        for p in mod.buffers():
            assert p.device == torch.device("cuda:0")

        dist = mod.build_distribution()
        assert dist.mean.device == torch.device("cuda:0")