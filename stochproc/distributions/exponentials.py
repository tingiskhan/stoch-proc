import torch
from pyro.distributions import (Exponential, ExponentialFamily,
                                TransformedDistribution, constraints,
                                transforms)
from torch.distributions.utils import broadcast_all


class NegativeExponential(TransformedDistribution):
    r"""
    Creates an Exponential distribution parameterized by :attr:`rate`.
    Example::
        >>> m = Exponential(-torch.tensor([1.0]))
        >>> m.sample() Exponential distributed with rate=-1
        tensor([ 0.1046])
    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    def __init__(self, rate, validate_args=None):
        base = Exponential(rate=rate, validate_args=validate_args)
        zeros, ones = torch.zeros_like(base.rate), torch.ones_like(base.rate)
        super().__init__(base, transforms.AffineTransform(zeros, -ones), validate_args=validate_args)

    @property
    def mean(self):
        return -self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    @property
    def stddev(self):
        return self.base_dist.stddev

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeExponential, _instance)
        batch_shape = torch.Size(batch_shape)

        new_base_dist = self.base_dist.expand(batch_shape)
        super(NegativeExponential, new).__init__(
            new_base_dist, transforms=self.transforms, validate_args=self._validate_args
        )
        return new


# TODO: Replace with two exponentials...?
class DoubleExponential(ExponentialFamily):
    r"""
    Creates a Double Exponential distribution parameterized by :attr:`rho_minus, rho_plus, p`.
    In particular:
        .. math::
            \zeta(q) = p \rho_{plus} \exp{-\rho_{plus} * q}, \text{if q > 0}
            \zeta(q) = p -\rho_{minus} \exp{-rho_{minus} * q}, \text{otherwise}

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution

    Example:
        >>> m = DoubleExponential(torch.tensor([10, -10, 0.5]))
        >>> m.sample()
        tensor([ 0.1046])

    """
    arg_constraints = {
        "p": constraints.interval(0.0, 1.0),
    }

    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.p / self.rho_plus + (1.0 - self.p) / self.rho_minus

    @property
    def stddev(self):
        # V(J) := E[J^2] - E[J]^2
        return (
            self.p * 2 * self._pos_exp.variance + (1 - self.p) * 2 * self._neg_exp.variance - self.mean.pow(2.0)
        ).sqrt()

    @property
    def variance(self):
        sigma = self.stddev
        return sigma.pow(2.0)

    @property
    def phi_fun(self):
        # eq. 30 in Hainaut&Moraux 2016
        # \phi(1, 0) =: c
        return self.p * (1.0 / self.rho_plus).exp() + (1 - self.p) * (1.0 / self.rho_minus).exp()

    @property
    def rho_plus(self) -> torch.Tensor:
        return self._pos_exp.rate

    @property
    def rho_minus(self) -> torch.Tensor:
        return -self._neg_exp.base_dist.rate

    def __init__(self, rho_plus, rho_minus, p, validate_args=None):
        """
        Internal initializer for :class:`DoubleExponential`.

        Args:
            rho_plus (_type_): rate of positive exponential.
            rho_minus (_type_): rate of negative exponential.
            p (_type_): probability of choosing negative exponential.
        """

        rho_plus, rho_minus, self.p = broadcast_all(rho_plus, rho_minus, p)

        self._pos_exp = Exponential(rho_plus, validate_args=False)
        self._neg_exp = NegativeExponential(rho_minus, validate_args=False)

        super().__init__(self._neg_exp.batch_shape, validate_args=validate_args)
        self.c = self.phi_fun

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DoubleExponential, _instance)
        batch_shape = torch.Size(batch_shape)

        new.p = self.p.expand(batch_shape)
        new._pos_exp = self._pos_exp.expand(batch_shape)
        new._neg_exp = self._neg_exp.expand(batch_shape)

        super(DoubleExponential, new).__init__(batch_shape, validate_args=self._validate_args)

        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.empty(shape, device=self.p.device).uniform_()

        x = torch.where(
            u < self.p,
            self._neg_exp.rsample(sample_shape),
            self._pos_exp.rsample(sample_shape),
        )

        return x

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_prob = torch.where(
            value >= 0.0, (1.0 - self.p) * self._pos_exp.log_prob(value), self.p * self._neg_exp.log_prob(value)
        )

        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        one_mp = 1.0 - self.p
        cdf = torch.where(
            value >= 0.0,
            one_mp + self.p * self._pos_exp.cdf(value),
            one_mp * self._neg_exp.cdf(value),
        )

        return cdf
