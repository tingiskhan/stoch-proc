import torch
from pyro.distributions import Delta, Poisson, Distribution
from ...timeseries import StochasticDifferentialEquation, TimeseriesState
from ...distributions import DistributionModule, JointDistribution, DoubleExponential

INFTY = float("inf")


class LambdaProcess(StochasticDifferentialEquation):
    """
    Implements a self exciting process as per the model implemented in [to be filled out] by Andrea Veronese.
    """

    def __init__(self, alpha_, xi_, eta_, p, rho_minus, rho_plus, **kwargs):
        """
        Initializes the ``LambdaProcess`` class.

        Args:
            alpha_: The speed of reversion.
            xi_: The location.
            eta_: The variance.
            p:
            rho_minus:
            rho_plus:
        """

        super().__init__(
            (alpha_, xi_, eta_),
            **kwargs
        )

        def _de(p_, rho_plus_, rho_minus_, **kwargs_):
            return DoubleExponential(p=p_, rho_plus=rho_plus_, rho_minus=-rho_minus_, **kwargs_)

        self.de = DistributionModule(_de, p_=p, rho_plus_=rho_plus, rho_minus_=rho_minus)

    @staticmethod
    def drift(x, alpha, xi, eta):
        return alpha * (xi - x)

    @staticmethod
    def diffusion(x, alpha, xi, eta):
        return eta

    def build_density(self, x: TimeseriesState) -> Distribution:
        r"""
        Joint density for the realizations of :math:`\lambda_t, dN_t, \lambda_s, q`.
        """

        alpha_, xi_, eta_ = self.functional_parameters()
        lambda_s = x.values[..., 0]

        dn_t = Poisson(rate=lambda_s * self.dt).sample()
        de = self.de.build_distribution().expand(lambda_s.shape)
        q = de.sample()

        deterministic = self.drift(lambda_s, alpha_, xi_, eta_) * self.dt

        diffusion = self.diffusion(lambda_s, alpha_, xi_, eta_) * q.abs() * dn_t
        lambda_t = (lambda_s + deterministic + diffusion).clip(0.0, INFTY)

        lambda_t[~torch.isfinite(lambda_t)] = lambda_s.max()

        return JointDistribution(Delta(lambda_t), Delta(dn_t), Delta(lambda_s), Delta(q))
