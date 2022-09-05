import torch
from pyro.distributions import Normal, MultivariateNormal
from torch.distributions.utils import broadcast_all
from torch.linalg import cholesky_ex

from ..chol_affine import LowerCholeskyAffineProcess
from ...distributions import DistributionModule
from ...typing import ParameterType


_info = torch.finfo(torch.get_default_dtype())
EPS = _info.eps


def _build_init(kappa, v_0, sigma, delta, x_0, sigma_x, lam, eta):
    loc = torch.stack((x_0, v_0), dim=-1)

    var_x = sigma_x.pow(2.0) / (2 * delta)

    kpd = kappa + delta
    kkd = kappa * kpd
    var_v = (
        sigma.pow(2.0) / (2.0 * kappa)
        + lam * eta * sigma * sigma_x / kkd
        + (lam * sigma_x).pow(2.0) / (2.0 * delta * kkd)
    )
    covariance = eta * sigma * sigma_x / kpd + lam * sigma_x.pow(2.0) / (2.0 * delta * kpd)

    cov_1 = torch.stack((var_x, covariance), dim=-1)
    cov_2 = torch.stack((covariance, var_v), dim=-1)
    cov = torch.stack((cov_1, cov_2), dim=-2)

    return MultivariateNormal(loc, cov)


class BivariateTrendingOU(LowerCholeskyAffineProcess):
    """
    Implements the (solved) bivariate `Trending OU process`_ of Chapter 4.

    .. _`Trending OU process`: https://deanstreetlab.github.io/papers/papers/Statistical%20Methods/Trending%20Ornstein-Uhlenbeck%20Process%20and%20its%20Applications%20in%20Mathematical%20Finance.pdf
    """

    def __init__(
        self,
        kappa: ParameterType,
        gamma: ParameterType,
        sigma: ParameterType,
        v_0: ParameterType,
        delta: ParameterType,
        sigma_x: ParameterType,
        eta: ParameterType,
        lamda: ParameterType = 1.0,
        dt: float = 1.0,
        **kwargs
    ):
        """
        Initializes the :class:`BivariateTrendingOU` class.

        Args:
            kappa: see :class:`Trending OU`
            gamma: see :class:`Trending OU`
            sigma: see :class:`Trending OU`
            v_0: see :class:`Trending OU`
            delta: reversion of the :math:`X` process.
            sigma_x: standard deviation of :math:`X` process.
            eta: correlation between :math:`q` and :math:`X`
            lamda: exposure to :math:`X` in :math:`q`
            dt: lamda: see :class:`Trending OU`
            **kwargs:
        """

        params = kappa, gamma, sigma, v_0, delta, x_0, sigma_x, eta, lamda = broadcast_all(
            kappa, gamma, sigma, v_0, delta, 0.0, sigma_x, eta, lamda
        )

        dist = DistributionModule(Normal, loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)
        initial_dist = DistributionModule(
            _build_init, kappa=kappa, v_0=v_0, sigma=sigma, delta=delta, x_0=x_0, sigma_x=sigma_x, lam=lamda, eta=eta
        )

        super(BivariateTrendingOU, self).__init__(self._mean_scale, params, initial_dist, dist, **kwargs)
        self._dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def _mean_scale(self, x, k, g, s, v_0, delta, x_0, s_x, eta, lamda):
        d_v = (-k * self._dt).exp()
        d_x = (-delta * self._dt).exp()

        # OU loc-var
        ou_loc = x_0 + (x.values[..., 0] - x_0) * d_x
        ou_var = s_x.pow(2.0) / (2.0 * delta) * (1.0 - d_x.pow(2.0))

        # Trend loc-var
        k_p_d = k + delta
        k_m_d = k - delta
        if k_m_d == 0.0:
            # TODO: Might cause issues...
            k_m_d.fill_(EPS)

        trend_loc = v_0 + g * (x.time_index + self._dt) + (x.values[..., 1] - g * x.time_index - v_0) * d_v
        loc = trend_loc + lamda / k_m_d * (d_x - d_v) * x.values[..., 0]

        d_v_2 = d_v.pow(2.0)
        d_x_2 = d_x.pow(2.0)
        var_1 = s.pow(2.0) / (2.0 * k) * (1.0 - d_v_2)

        # TODO: Add the auxiliary variables for 1 - d_x etc.
        d_v_x = d_v * d_x
        var_2_1 = 2.0 * lamda * eta * s * s_x / k_m_d
        var_2_2 = (1.0 - d_v_x) / k_p_d + (d_v_2 - 1.0) / 2.0 / k

        var_3_1 = (lamda * s_x / k_m_d).pow(2.0)
        var_3_2 = (1.0 - d_x_2) / 2.0 / delta + 2.0 * (d_v_x - 1.0) / k_p_d + (1.0 - d_v_2) / 2.0 / k

        var = var_1 + var_2_1 * var_2_2 + var_3_1 * var_3_2

        # Covar
        covar_1 = eta * s * s_x / k_p_d * (1.0 - d_v_x)
        covar_2 = lamda * s_x.pow(2.0) / (2.0 * delta * k_p_d)
        covar_3 = (k * (1.0 - d_x_2) - delta * (1.0 - 2.0 * d_v_x + d_x_2)) / k_m_d

        covar = covar_1 + covar_2 * covar_3

        # Stack
        loc = torch.stack((ou_loc, loc), dim=-1)
        cov_1 = torch.stack((ou_var, covar), dim=-1)
        cov_2 = torch.stack((covar, var), dim=-1)
        cov = torch.stack((cov_1, cov_2), dim=-2)

        return loc, cholesky_ex(cov)[0]
