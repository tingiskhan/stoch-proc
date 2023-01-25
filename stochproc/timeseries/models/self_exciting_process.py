from pyro.distributions import Delta, Bernoulli, Distribution, TransformedDistribution, transforms as t, Normal
import torch

from ...distributions import JointDistribution, DoubleExponential
from ..diffusion import StochasticDifferentialEquation
from ..state import TimeseriesState
from ...typing import ParameterType


class SelfExcitingLatentProcesses(StochasticDifferentialEquation):
    """
    Class defining the process for the istantaneous frequency of jumps, where jumps are distributed as a double exponential r.v.
    See e.g. https://www.researchgate.net/publication/327672329_Hedging_of_options_in_presence_of_jump_clustering
    """

    def __init__(
        self,
        alpha: ParameterType,
        xi: ParameterType,
        eta: ParameterType,
        p: ParameterType,
        rho_minus: ParameterType,
        rho_plus: ParameterType,
        **kwargs
    ):
        """
        Internal initializer for :class:`SelfExcitingLatentProcesses`.

        Args:
            alpha (ParameterType): speed of mean reversion for deterministic component.
            xi (ParameterType): level to which to revert to.
            eta (ParameterType): volatility of volatility.
            p (ParameterType): _description_
            rho_minus (ParameterType): _description_
            rho_plus (ParameterType): _description_
        """

        super().__init__(self.kernel, (alpha, xi, eta), initial_kernel=None, **kwargs)
        self.de = DoubleExponential(p=p, rho_plus=rho_plus, rho_minus=rho_minus)
        
        self._initial_kernel = self.initial_kernel
        self._event_shape = torch.Size([4])

    def initial_kernel(self, alpha, xi, eta):
        exp_j2 = self.de.p * 2.0 * self.de.rho_plus.pow(-2.0) + (1.0 - self.de.p) * 2.0 * self.de.rho_minus.pow(-2.0)

        std_lambda = exp_j2.sqrt() * xi
        dist_ = TransformedDistribution(
            Normal(torch.zeros_like(alpha), torch.ones_like(alpha)),
            [t.AffineTransform(xi, std_lambda), t.AbsTransform()],
        )

        return JointDistribution(dist_, Delta(torch.zeros_like(alpha)), Delta(torch.zeros_like(alpha)), self.de)

    def kernel(self, x: TimeseriesState, alpha, xi, eta) -> Distribution:
        r"""
        Joint density for the realizations of (\lambda_t, dN_t, \lambda_s, q).
        N.B.: Jumps are modeled as a Bernoulli random variable (there could at most one jump for each dt).
        """

        lambda_s = x.value[..., 0]

        # TODO: Use unbound?
        probs = (lambda_s * self.dt).nan_to_num(0.0, 0.0, 0.0).clip(0.0, 1.0)
        dn_t = Bernoulli(probs=probs).sample()

        de = self.de.expand(lambda_s.shape)
        dl_t = de.sample() * dn_t

        deterministic = alpha * (xi - lambda_s) * self.dt

        diffusion = eta * dl_t.abs()
        max_lambda = 1000.0
        lambda_t = (lambda_s + deterministic + diffusion).clip(0.0, max_lambda)

        return JointDistribution(Delta(lambda_t), Delta(dn_t), Delta(lambda_s), Delta(dl_t))
