import torch
from pyro.distributions import Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..affine import AffineProcess


def f(x, sigma_volatility):
    x1 = x.value[..., 1].exp()
    x2 = sigma_volatility

    return x.value, torch.stack(torch.broadcast_tensors(x1, x2), dim=-1)


def initial_kernel(loc, sigma_volatility):
    return Normal(loc=loc, scale=sigma_volatility.unsqueeze(-1)).to_event(1)


class UCSV(AffineProcess):
    r"""
    Implements a UCSV model, i.e. a stochastic process with the dynamics
        .. math::
            L_{t+1} = L_t + V_t W_{t+1}, \newlin
            \log{V_{t+1}} = \log{V_t} + \sigma_v U_{t+1}, \newlin
            L_0, \log{V_0} \sim \mathcal{N}(x^i_0, \sigma^v), \: i \in [L, V].

    where :math:`\sigma_v > 0`.
    """

    def __init__(self, sigma_volatility: ParameterType, initial_state_mean: ParameterType = torch.zeros(2)):
        r"""
        Inititalizes :class:`UCSV`.

        Args:
            sigma_volatility: The volatility of the log volatility process, i.e. :math:`\sigma_v`.
            initial_state_mean: Optional, whether to use initial values other than 0 for both processes.
            kwargs: See base.
        """

        sigma_volatility = broadcast_all(sigma_volatility)[0]

        increment_dist = (
            Normal(
                loc=torch.tensor(0.0, device=sigma_volatility.device),
                scale=torch.tensor(1.0, device=sigma_volatility.device),
            )
            .expand(sigma_volatility.shape + torch.Size([2]))
            .to_event(1)
        )

        super().__init__(f, (sigma_volatility,), increment_dist, initial_kernel, (initial_state_mean, sigma_volatility))
