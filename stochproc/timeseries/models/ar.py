from functools import partial

import torch
from pyro.distributions import (Delta, Distribution, Normal,
                                TransformedDistribution)
from pyro.distributions.transforms import AffineTransform
from torch.distributions.utils import broadcast_all

from ...distributions import JointDistribution
from ..linear import LinearModel


# TODO: Add beta for those where abs(beta) < 1.0
def _initial_kernel(a, b, s, lags):
    alpha = b
    beta = a
    sigma = s

    if lags > 1:
        alpha = alpha[..., 0]
        beta = beta[..., 0, 0]
        sigma = sigma[..., 0]

    loc = alpha
    scale = sigma / (1.0 - beta.pow(2.0)).sqrt()

    scale = torch.where(beta.abs() < 1.0, scale, sigma)
    base = _build_trans_dist(torch.zeros_like(loc), torch.ones_like(scale), lags)

    if lags > 1:
        loc.unsqueeze_(-1)
        scale.unsqueeze_(-1)

    # NB: As we utilize delta for lags, we can use same loc/scale
    return TransformedDistribution(base, AffineTransform(loc, scale))


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
        Internal initializer for :class:`AR`.

        Args:
            alpha: mean of the process.
            beta: reversion of the process, usually constrained to :math:`(-1, 1)`.
            sigma: volatility of the process.
            lags: number of lags.
            kwargs: see base.
        """

        alpha, sigma = broadcast_all(alpha, sigma)
        (beta,) = broadcast_all(beta)

        if (lags > 1) and (beta.shape[-1] != lags):
            raise Exception(f"Mismatch between shapes: {alpha.value.shape[-1]} != {lags}")

        self.lags = lags
        inc_dist = _build_trans_dist(
            loc=torch.tensor(0.0, device=alpha.device), scale=torch.tensor(1.0, device=alpha.device), lags=self.lags
        )

        # Create parameters
        a, b, s = beta, alpha, sigma
        if self.lags > 1:
            bottom_shape = self.lags - 1, self.lags
            bottom = torch.eye(*bottom_shape, device=beta.device).expand(beta.shape[:-1] + bottom_shape)

            b_masker = (
                torch.eye(self.lags, 1, device=alpha.device)
                .squeeze(-1)
                .expand(beta.shape[:-1] + torch.Size([self.lags]))
            )

            a = torch.cat((a.unsqueeze(-2), bottom), dim=-2)
            b = b_masker * b.unsqueeze(-1)
            s = b_masker * s.unsqueeze(-1)

        super().__init__(
            (a, b, s),
            increment_distribution=inc_dist,
            initial_kernel=partial(_initial_kernel, lags=self.lags),
        )

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        new = self._get_checked_instance(AR)

        super(AR, new).__init__(
            new_parameters["parameters"],
            self.increment_distribution,
            self._initial_kernel,
            new_parameters["initial_parameters"],
        )
        new.lags = self.lags

        return new
