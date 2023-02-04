import torch
from pyro.distributions import Normal, Delta
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..affine import AffineProcess


def _f(x, sigma_volatility):
    x1 = x.value[..., 1].exp()
    x2 = sigma_volatility

    return x.value, torch.stack(torch.broadcast_tensors(x1, x2), dim=-1)


def _initial_kernel(loc):
    return Delta(loc, event_dim=1)


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
        Internal initializer for :class:`UCSV`.

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
            .expand(torch.Size([2]))
            .to_event(1)
        )

        super().__init__(_f, (sigma_volatility,), increment_dist, _initial_kernel, (initial_state_mean,))

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        return UCSV(new_parameters["parameters"][0], new_parameters["initial_parameters"][0])
