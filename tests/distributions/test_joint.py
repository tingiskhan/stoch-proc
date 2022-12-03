import pytest
import torch
from pyro.distributions import (
    Exponential,
    StudentT,
    Independent,
    TransformedDistribution,
)
from pyro.distributions.transforms import AffineTransform, ExpTransform

from stochproc.torch.distributions import JointDistribution


@pytest.fixture
def joint_distribution():
    dist_1 = Exponential(rate=1.0)
    dist_2 = Independent(StudentT(df=torch.ones(2)), 1)

    return JointDistribution(dist_1, dist_2)


@pytest.fixture
def joint_distribution_inverted(joint_distribution):
    return JointDistribution(*joint_distribution.distributions[::-1], *joint_distribution.distributions)


class TestJointDistribution(object):
    def test_mask(self, joint_distribution):
        assert joint_distribution.indices[0] == 0
        assert joint_distribution.indices[1] == slice(1, 3)

        assert joint_distribution.event_shape == torch.Size([3])

    def test_samples(self, joint_distribution):
        samples = joint_distribution.sample()

        assert samples.shape == torch.Size([3])

        more_sample_shape = torch.Size([1000, 300])
        more_samples = joint_distribution.sample(more_sample_shape)

        assert more_samples.shape == torch.Size([*more_sample_shape, 3])

    def test_log_prob(self, joint_distribution):
        shape = torch.Size([1000, 300])

        samples = joint_distribution.sample(shape)
        log_prob = joint_distribution.log_prob(samples)

        assert log_prob.shape == shape

    def test_entropy(self, joint_distribution):
        expected = joint_distribution.distributions[0].entropy() + joint_distribution.distributions[1].entropy()
        assert joint_distribution.entropy() == expected

    def test_gradient(self, joint_distribution):
        samples = joint_distribution.sample()

        joint_distribution.distributions[0].rate.requires_grad_(True)
        log_prob = joint_distribution.log_prob(samples)

        assert log_prob.requires_grad

        log_prob.backward()
        assert joint_distribution.distributions[0].rate.grad is not None

    def test_expand(self, joint_distribution):
        new_shape = torch.Size([1000, 10])
        expanded = joint_distribution.expand(new_shape)

        assert expanded.batch_shape == new_shape

    def test_transform(self, joint_distribution):
        transformed = TransformedDistribution(joint_distribution, AffineTransform(0.0, 0.0))
        samples = transformed.sample()

        assert (samples == 0.0).all()

        mean = 1.0
        transformed = TransformedDistribution(joint_distribution, AffineTransform(mean, 1.0))
        samples = transformed.sample()

        assert transformed.log_prob(samples) == joint_distribution.log_prob(samples - mean)

        exp_transform = TransformedDistribution(joint_distribution, ExpTransform())
        samples = exp_transform.sample((1000,))

        assert (samples >= 0.0).all()

    def test_joint_distribution_mask_1(self, joint_distribution_inverted):
        assert joint_distribution_inverted.indices[0] == slice(0, 2)
        assert joint_distribution_inverted.indices[1] == 2

        assert joint_distribution_inverted.indices[2] == 3
        assert joint_distribution_inverted.indices[3] == slice(4, 6)

    def test_joint_distribution_different_batch_shapes(self):
        shape = [10]

        dist_1 = Exponential(rate=torch.ones(shape + [1, 5]))
        dist_2 = StudentT(3.0 * torch.ones(shape + [5, 5]))

        joint = JointDistribution(dist_1, dist_2)

        assert joint.batch_shape == dist_2.batch_shape

