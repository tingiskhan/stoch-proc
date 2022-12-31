from functools import partial

import torch
from pyro.distributions import Delta, Distribution, Normal, TransformedDistribution
from pyro.distributions.transforms import AffineTransform
from torch.distributions.utils import broadcast_all

from ...distributions import JointDistribution
from ..linear import LinearModel


# TODO: Add beta for those where abs(beta) < 1.0
def _initial_kernel(alpha, beta, sigma, lags):
    base = _build_trans_dist(torch.zeros_like(beta), torch.ones_like(beta), lags)

    return TransformedDistribution(
        base, AffineTransform(beta.unsqueeze(-1) if lags > 1 else beta, sigma.unsqueeze(-1) if lags > 1 else sigma)
    )


def _build_trans_dist(loc, scale, lags) -> Distribution:
    base = Normal(loc=loc, scale=scale)
    if lags == 1:
        return base

    zeros = torch.zeros((*loc.shape, lags - 1), device=loc.device)
    return JointDistribution(base, Delta(zeros, event_dim=1))


class AR(LinearModel):
    r"""
    Implements an AR(k) process, i.e. a process given by
        .. math::
            X_{t+1} = \alpha + \beta X_t + \sigma W_t, \newline
            X_0 \sim \mathcal{N}(\alpha, \frac{\sigma}{\sqrt{1 - \beta^2}},

    where :math:`W_t` is a univariate zero mean, unit variance Gaussian random variable.
    """

    def __init__(self, alpha, beta, sigma, lags=1):
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
        inc_dist = _build_trans_dist(
            loc=torch.tensor(0.0, device=alpha.device), scale=torch.tensor(1.0, device=alpha.device), lags=self.lags
        )

        bottom_shape = self.lags - 1, self.lags

        self._bottom = torch.eye(*bottom_shape, device=alpha.device)
        self._b_masker = torch.eye(self.lags, 1, device=alpha.device).squeeze(-1)

        super().__init__(
            (beta, sigma, alpha),
            increment_distribution=inc_dist,
            initial_kernel=partial(_initial_kernel, lags=self.lags),
            parameter_transform=self._param_transform
        )        

    def _param_transform(self, a, b, s):
        if self.lags == 1:
            return a, b, s
        
        batch_shape = a.shape[:-1]

        mask = torch.ones((*batch_shape, *self._bottom.shape), device=a.device)
        bottom = self._bottom * mask

        a = torch.cat((a.unsqueeze(-2), bottom), dim=-2)
        b = self._b_masker * b

        return a, b, s.unsqueeze(-1)
