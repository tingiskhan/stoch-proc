import torch
from torch.distributions import Normal
from ..linear import LinearModel
from ...typing import ParameterType
from ...distributions import DistributionModule
from ...utils import enforce_named_parameter


class LocalLinearTrend(LinearModel):
    r"""
    Implements a Local Linear Trend model, i.e. a model with the following dynamics
        .. math::
            L_{t+1} = L_t + S_t + \sigma_l W_{t+1}, \newline
            S_{t+1} = S_t + \sigma_s V_{t+1},

    where :math:`\sigma_i > 0``, and :math:`W_t, V_t` are two independent zero mean and unit variance Gaussians.
    """

    def __init__(
        self,
        sigma: ParameterType,
        initial_mean: ParameterType = torch.zeros(2),
        **kwargs
    ):
        r"""
        Initializes the :class:`LocalLinearTrend` class.

        Args:
            sigma: the vector :math:`[ \sigma_s, \sigma_l ]`.
            initial_mean: the initial mean.
            kwargs: see base.
        """

        sigma = enforce_named_parameter(scale=sigma)[0]

        increment_dist = DistributionModule(
            Normal, loc=torch.zeros(2), scale=torch.ones(2), reinterpreted_batch_ndims=1
        )

        initial_dist = DistributionModule(Normal, loc=initial_mean, scale=sigma, reinterpreted_batch_ndims=1)
        a = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

        super().__init__(a, sigma, increment_dist, initial_dist=initial_dist, **kwargs)
