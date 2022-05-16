import torch
from torch.distributions import Normal, Independent
from ...typing import ParameterType
from ...distributions import DistributionModule
from ..affine import AffineProcess
from ...utils import enforce_named_parameter


def f(x, sigma_volatility):
    x1 = x.values[..., 1].exp()
    x2 = sigma_volatility

    return x.values, torch.stack(torch.broadcast_tensors(x1, x2), dim=-1)


def _init_builder(loc, sigma_volatility):
    return Independent(Normal(loc=loc, scale=sigma_volatility), 1)


class UCSV(AffineProcess):
    r"""
    Implements a UCSV model, i.e. a stochastic process with the dynamics
        .. math::
            L_{t+1} = L_t + V_t W_{t+1}, \n
            \log{V_{t+1}} = \log{V_t} + \sigma_v U_{t+1}, \n
            L_0, \log{V_0} \sim \mathcal{N}(x^i_0, \sigma^v), \: i \in [L, V].

    where :math:`\sigma_v > 0`.
    """

    def __init__(
        self,
        sigma_volatility: ParameterType,
        initial_state_mean: ParameterType = torch.zeros(2),
        **kwargs
    ):
        """
        Inititalizes :class:`UCSV`.

        Args:
            sigma_volatility: The volatility of the log volatility process, i.e. :math:`\\sigma_v`.
            initial_state_mean: Optional, whether to use initial values other than 0 for both processes.
            kwargs: See base.
        """

        sigma_volatility = enforce_named_parameter(sigma_volatility=sigma_volatility)[0]
        initial_dist = DistributionModule(_init_builder, loc=initial_state_mean, sigma_volatility=sigma_volatility)

        increment_dist = DistributionModule(
            Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1
        )

        super().__init__(f, (sigma_volatility,), initial_dist, increment_dist, **kwargs)
