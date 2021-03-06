import torch
from pyro.distributions import Delta, Normal, TransformedDistribution, Distribution
from pyro.distributions.transforms import AffineTransform

from torch.distributions.utils import broadcast_all

from ..linear import LinearModel
from ...distributions import DistributionModule, JointDistribution


# TODO: Add beta for those where abs(beta) < 1.0
def _build_init(alpha, beta, sigma, lags, **kwargs):
    base = _build_trans_dist(0.0, 1.0, lags, **kwargs)

    return TransformedDistribution(base, AffineTransform(alpha, sigma))


def _build_trans_dist(loc, scale, lags, **kwargs) -> Distribution:
    base = Normal(loc=loc, scale=scale, **kwargs)
    if lags == 1:
        return base

    return JointDistribution(base, Delta(torch.zeros(lags - 1), event_dim=1), **kwargs)


class AR(LinearModel):
    r"""
    Implements an AR(k) process, i.e. a process given by
        .. math::
            X_{t+1} = \alpha + \beta X_t + \sigma W_t, \newline
            X_0 \sim \mathcal{N}(\alpha, \frac{\sigma}{\sqrt{1 - \beta^2}},

    where :math:`W_t` is a univariate zero mean, unit variance Gaussian random variable.
    """

    def __init__(self, alpha, beta, sigma, lags=1, **kwargs):
        """
        Initializes the :class:`AR` class.

        Args:
            alpha: mean of the process.
            beta: reversion of the process, usually constrained to :math:`(-1, 1)`.
            sigma: volatility of the process.
            lags: number of lags.
            kwargs: see base.
        """

        alpha, sigma = broadcast_all(alpha, sigma)
        beta = broadcast_all(beta)[0]

        if (lags > 1) and (beta.shape[-1] != lags):
            raise Exception(f"Mismatch between shapes: {alpha.value.shape[-1]} != {lags}")

        self.lags = lags
        inc_dist = DistributionModule(_build_trans_dist, loc=0.0, scale=1.0, lags=self.lags)
        initial_dist = DistributionModule(_build_init, alpha=alpha, beta=beta, sigma=sigma, lags=self.lags)

        super().__init__(beta, sigma, increment_dist=inc_dist, b=alpha, initial_dist=initial_dist, **kwargs)
        self.mean_scale_fun = self._mean_scale_wrapper(self.mean_scale_fun)

        bottom_shape = self.lags - 1, self.lags
        self.register_buffer("_bottom", torch.eye(*bottom_shape))
        self.register_buffer("_b_masker", torch.eye(self.lags, 1).squeeze(-1))

    def _mean_scale_wrapper(self, f):
        def _wrapper(x, a, b, s):
            if self.lags == 1:
                return f(x, a, b, s)

            batch_shape = a.shape[:-1]

            mask = torch.ones((*batch_shape, *self._bottom.shape), device=a.device)
            bottom = self._bottom * mask

            a = torch.cat((a.unsqueeze(-2), bottom))
            b = self._b_masker * b

            return f(x, a, b, s)

        return _wrapper
