import pytest
import torch
from torch.distributions import Exponential, StudentT, Independent, Normal

from stochproc.distributions import Prior, DistributionModule


@pytest.fixture
def dist_with_prior():
    prior = Prior(Exponential, rate=0.1)
    return DistributionModule(StudentT, df=prior)


class TestPrior(object):
    def test_prior(self, dist_with_prior):
        dist = dist_with_prior()

        assert (
                isinstance(dist, StudentT) and
                (dist.df > 0).all() and
                (dist.df == dist_with_prior().df) and
                len(tuple(dist_with_prior.parameters_and_priors())) == 1
        )

    def test_prior_sample_parameter(self, dist_with_prior):
        size = torch.Size([1000])

        for parameter, prior in dist_with_prior.parameters_and_priors():
            parameter.sample_(prior, size)

        dist = dist_with_prior()

        assert dist.df.shape == size

    def test_prior_multivariate(self):
        loc = torch.zeros(3)
        scale = torch.ones(3)

        prior = Prior(Normal, loc=loc, scale=scale, reinterpreted_batch_ndims=1)

        dist = prior.build_distribution()
        assert (
                isinstance(dist, Independent) and
                isinstance(dist.base_dist, Normal) and
                (dist.reinterpreted_batch_ndims == 1) and
                (dist.base_dist.loc == loc).all() and
                (dist.base_dist.scale == scale).all()
        )

