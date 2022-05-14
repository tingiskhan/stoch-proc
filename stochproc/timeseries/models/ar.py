from torch.distributions import Normal, TransformedDistribution, AffineTransform, Distribution
from pyro.distributions import Delta
import torch
from ..linear import LinearModel
from ...distributions import DistributionModule, JointDistribution
from ...utils import enforce_named_parameter


# TODO: Add beta for those where abs(beta) < 1.0
def _build_init(alpha, beta, sigma, lags, **kwargs):
    base = _build_trans_dist(0.0, 1.0, lags, **kwargs)
    std = sigma

    return TransformedDistribution(base, AffineTransform(alpha, std.sqrt()))


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

        alpha, beta, sigma = enforce_named_parameter(alpha=alpha, beta=beta, sigma=sigma)

        if (lags > 1) and (beta.value.shape[-1] != lags):
            raise Exception(f"Mismatch between shapes: {alpha.value.shape[-1]} != {lags}")

        self.lags = lags
        inc_dist = DistributionModule(_build_trans_dist, loc=0.0, scale=1.0, lags=self.lags)
        initial_dist = DistributionModule(_build_init, alpha=alpha, beta=beta, sigma=sigma, lags=self.lags)

        def _parameter_transform(a, b, s):
            if self.lags == 1:
                return a, b, s

            batch_shape = a.shape[:-1]

            bottom_shape = self.lags - 1, self.lags
            mask = torch.ones((*batch_shape, *bottom_shape), device=a.device)
            bottom = torch.eye(*bottom_shape, device=a.device) * mask

            mat = torch.cat((a.unsqueeze(-2), bottom))

            return mat, b, s

        super().__init__(beta, sigma, increment_dist=inc_dist, b=alpha, initial_dist=initial_dist, parameter_transform=_parameter_transform, **kwargs)
