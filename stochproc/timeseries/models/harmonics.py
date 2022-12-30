from pyro.distributions import Normal
import torch

from ..linear import LinearModel
from ..utils import coerce_tensors
from ...typing import ParameterType


def _parameter_transform(lamda, s):
    cos_lam = torch.cos(lamda)
    sin_lam = torch.sin(lamda)

    a_top = torch.stack([cos_lam, sin_lam], dim=-1)
    a_bottom = torch.stack([-sin_lam, cos_lam], dim=-1)

    a = torch.stack([a_top, a_bottom], dim=-2)

    return a, torch.zeros_like(s), s


def initial_kernel(x0, s):
    return Normal(x0, s).to_event(1)


class HarmonicProcess(LinearModel):
    r"""
    Implements a harmonic timeseries process of the form
        .. math::
            \gamma_{t + 1} = \gamma \cos{ \lambda } + \gamma^*\sin{ \lambda } + \sigma \nu_{t + 1}, \newline
            \gamma^*_{t + 1} = -\gamma \sin { \lambda } + \gamma^* \cos{ \lambda } + \sigma^* \nu^*_{t + 1}.
    """

    def __init__(self, lamda: ParameterType, sigma: ParameterType, x_0: ParameterType = None):
        """
        Internal initializer for :class:`HarmonicProcess`.

        Args:
            lamda (ParameterType): coefficient for periodic component.
            sigma (ParameterType): the st
        """

        lamda, sigma = coerce_tensors(lamda, sigma)

        if x_0 is None:
            x_0 = torch.zeros(2, device=lamda.device)

        lamda, sigma, x_0 = coerce_tensors(lamda, sigma, x_0)
        increment_distribution = Normal(
            torch.zeros(2, device=lamda.device), torch.ones(2, device=lamda.device)
        ).to_event(1)

        initial_parameters = (x_0, sigma)

        super().__init__(
            (lamda, sigma),
            increment_distribution,
            initial_kernel,
            initial_parameters=initial_parameters,
            parameter_transform=_parameter_transform
        )
