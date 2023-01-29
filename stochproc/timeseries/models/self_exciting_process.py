from functools import partial
from pyro.distributions import Delta, Poisson, Distribution, TransformedDistribution, transforms as t, Normal
import torch

from ...distributions import JointDistribution, DoubleExponential
from ..diffusion import StochasticDifferentialEquation
from ..state import TimeseriesState
from ...typing import ParameterType


def _initial_kernel(alpha, xi, _, de):
    exp_j2 = de.p * 2.0 * de.rho_plus.pow(-2.0) + (1.0 - de.p) * 2.0 * de.rho_minus.pow(-2.0)

    std_lambda = exp_j2.sqrt() * xi
    dist_ = TransformedDistribution(
        Normal(torch.zeros_like(alpha), torch.ones_like(alpha)),
        [t.AffineTransform(xi, std_lambda), t.AbsTransform()],
    )

    return JointDistribution(dist_, Delta(torch.zeros_like(std_lambda)))


class SelfExcitingLatentProcesses(StochasticDifferentialEquation):
    """
    Class defining the process for the instantaneous frequency of jumps, where jumps are distributed as a double 
    exponential r.v. See `this_` paper for example.
    
    .. _`this`: https://www.researchgate.net/publication/327672329_Hedging_of_options_in_presence_of_jump_clustering
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

        self.de = DoubleExponential(p=p, rho_plus=rho_plus, rho_minus=rho_minus)
        init_kernel = partial(_initial_kernel, de=self.de)

        super().__init__(self.kernel, (alpha, xi, eta), initial_kernel=init_kernel, **kwargs)

    def kernel(self, x: TimeseriesState, alpha, xi, eta) -> Distribution:
        r"""
        Joint density for the realizations of :math:`(\lambda_t, dN_t, \lambda_s, q)`.
        """

        lambda_s = x.value[..., 0]

        intensity = (lambda_s * self.dt).nan_to_num(0.0, 0.0, 0.0).clip(min=0.0)
        dn_t = Poisson(rate=intensity).sample()

        de = self.de.expand(lambda_s.shape)
        dl_t = de.sample() * dn_t

        deterministic = alpha * (xi - lambda_s) * self.dt

        diffusion = eta * dl_t.abs()
        lambda_t = (lambda_s + deterministic + diffusion).clip(min=0.0)

        combined = torch.stack((lambda_t, dl_t), dim=-1)

        return Delta(combined, event_dim=1)
