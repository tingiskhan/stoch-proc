from torch.distributions import ExponentialFamily, constraints, Exponential
from torch.distributions.utils import broadcast_all

from numbers import Number
import torch


class NegativeExponential(ExponentialFamily):
    r"""
    Creates an Exponential distribution parameterized by :attr:`rate`.
    Example::
        >>> m = Exponential(-torch.tensor([1.0]))
        >>> m.sample() Exponential distributed with rate=-1
        tensor([ 0.1046])
    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    arg_constraints = {"rate": constraints.less_than(0.0)}
    support = constraints.less_than(0.0)
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.rate.reciprocal()

    @property
    def stddev(self):
        return -self.rate.reciprocal()

    @property
    def variance(self):
        return self.rate.pow(-2)

    def __init__(self, rate, validate_args=None):
        (self.rate,) = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, Number) else self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super().__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for ._exponential()
            u = torch.rand(shape, dtype=self.rate.dtype, device=self.rate.device)
            return (-u).log1p() / self.rate
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (-self.rate).log() - self.rate * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        return -torch.log(1 - value) / self.rate

    def entropy(self):
        return 1.0 - torch.log(-self.rate)

    @property
    def _natural_params(self):
        return (-self.rate,)

    def _log_normalizer(self, x):
        return -torch.log(-x)


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
        "rho_minus": constraints.less_than(0.0),
        "p": constraints.interval(0.0, 1.0),
    }

    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.p / self.rho_plus + (1 - self.p) / self.rho_minus

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
        c = self.p * (1 / self.rho_plus).exp() + (1 - self.p) * (1 / self.rho_minus).exp()
        return c

    def __init__(self, rho_plus, rho_minus, p, validate_args=None):
        (
            self.rho_plus,
            self.rho_minus,
            self.p,
        ) = broadcast_all(rho_plus, rho_minus, p)

        if isinstance(rho_plus, Number) and isinstance(rho_minus, Number) and isinstance(p, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.p.size()

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
        x = torch.zeros(size=shape, device=self.p.device)
        u = torch.rand(size=shape, device=self.p.device)

        mask = u < self.p
        not_mask = ~mask

        x[mask] = -1.0 / self.rho_minus[mask] * (u[mask] / (1.0 - self.p[mask])).log()
        x[not_mask] = -1.0 / self.rho_plus[~mask] * ((1.0 - u[not_mask]) / self.p[not_mask]).log()

        return x

    # TODO: Speed up if necessary
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        mask = (value >= 0.0).float()
        log_prob = ((self.p * self.rho_plus).log() - self.rho_plus * value) * mask + (
            (-self.rho_minus * (1 - self.p)).log() - self.rho_minus * value
        ) * (~mask)

        return log_prob

    # TODO: Speed up if necessary
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = value.size()
        u = torch.zeros(n)
        for i in range(0, n.__getitem__(0)):
            element = value[i]
            if element >= 0:
                u[i] = (1 - self.p) + self.p * (1 - torch.exp(-self.rho_plus * element))
            else:
                u[i] = (1 - self.p) * (1 - torch.exp(-self.rho_minus * element))
        return u

    def icdf(self, value):
        n = value.size()
        u = torch.zeros(n)
        x = torch.zeros(n)
        for i in range(0, n.__getitem__(0)):
            if u[i] <= self.p:
                x[i] = -1 / self.rho_minus * (u[i] / (1 - self.p)).log()
            else:
                x[i] = -1 / self.rho_plus * ((1 - u[i]) / self.p).log()
        return x

    def entropy(self):
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self):
        print(" I am in natural parameters")
        return (-self.rate,)

    def _log_normalizer(self, x):
        print(" I am in log_normalized")
        return -torch.log(-x)
