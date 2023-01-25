from pyro.distributions import ExponentialFamily, constraints, Exponential, TransformedDistribution, transforms
from torch.distributions.utils import broadcast_all

from numbers import Number
import torch


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
        "rho_plus": constraints.positive,
        "rho_minus": constraints.positive,
        "p": constraints.interval(0.0, 1.0),
    }

    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.p / self.rho_plus - (1 - self.p) / self.rho_minus

    @property
    def stddev(self):
        # V(J) := E[J^2] - E[J]^2
        return (
            self.p * 2 * self.rho_plus.pow(-2.0) + (1 - self.p) * 2 * self.rho_minus.pow(-2.0) - self.mean.pow(2.0)
        ).sqrt()

    @property
    def variance(self):
        sigma = self.stddev
        return sigma.pow(2.0)

    @property
    def phi_fun(self):
        # eq. 30 in Hainaut&Moraux 2016
        # \phi(1, 0) =: c
        c = self.p * (1 / self.rho_plus).exp() - (1 - self.p) * (1 / self.rho_minus).exp()
        return c

    def __init__(self, rho_plus, rho_minus, p, validate_args=None):
        (
            self.rho_plus,
            self.rho_minus,
            self.p,
        ) = broadcast_all(rho_plus, rho_minus, p)

        batch_shape = self.rho_minus.shape
        super().__init__(batch_shape, validate_args=validate_args)
        self.c = self.phi_fun

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DoubleExponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.p = self.p.expand(batch_shape)
        new.rho_plus = self.rho_plus.expand(batch_shape)
        new.rho_minus = self.rho_minus.expand(batch_shape)

        super(DoubleExponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.empty(shape, device=self.p.device).uniform_()

        x = torch.where(
            u < self.p, 
            1.0 / self.rho_minus * (u / (1.0 - self.p)).log(), 
            -1.0 / self.rho_plus * ((1.0 - u) / self.p).log()
        )

        return x

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_prob = torch.where(
            value >= 0.0,
            (self.p * self.rho_plus).log() - self.rho_plus * value,
            (self.rho_minus * (1 - self.p)).log() + self.rho_minus * value
        )

        return log_prob
    
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)

        cdf = torch.where(
            value >= 0.0,
            (1 - self.p) + self.p * (1 - torch.exp(-self.rho_plus * value)),
            (1 - self.p) * (1 - torch.exp(self.rho_minus * value))
        )

        return cdf

    def entropy(self):
        return 1.0 - torch.log(self.rate)
