from torch.distributions import Normal, TransformedDistribution, AffineTransform, Distribution
from pyro.distributions import Delta
import torch
from ..linear import LinearModel
from ...distributions import DistributionWrapper, JointDistribution


def _init_trans(module: "AR", dist):
    beta, alpha, sigma = module.functional_parameters()
    return TransformedDistribution(dist, AffineTransform(alpha, sigma / (1 - beta ** 2).sqrt()))


def _build_trans_dist(loc, scale, lags, **kwargs) -> Distribution:
    base = Normal(loc=loc, scale=scale, **kwargs)
    if lags == 1:
        return base

    return JointDistribution(base, Delta(torch.zeros(lags - 1), event_dim=1), **kwargs)


class AR(LinearModel):
    """
    Implements an AR(k) process, i.e. a process given by
        .. math::
            X_{t+1} = \\alpha + \\beta X_t + \\sigma W_t, \n
            X_0 \\sim \\mathcal{N}(\\alpha, \\frac{\\sigma}{\\sqrt{(1 - \\beta^2)})
    """

    def __init__(self, alpha, beta, sigma, lags=1, **kwargs):
        """
        Initializes the ``AR`` class.

        Args:
            alpha: The mean of the process.
            beta: The reversion of the process, usually constrained to :math:`(-1, 1)`.
            sigma: The volatility of the process.
            lags: The number of lags.
            kwargs: See base.
        """

        if (lags > 1) and (beta.shape[-1] != lags):
            raise Exception(f"Mismatch between shapes: {alpha.shape[-1]} != {lags}")

        self.lags = lags
        inc_dist = DistributionWrapper(_build_trans_dist, loc=0.0, scale=1.0, lags=lags)
        super().__init__(beta, sigma, increment_dist=inc_dist, b=alpha, initial_transform=_init_trans, **kwargs)
