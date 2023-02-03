import torch
from pyro.distributions import Delta, Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..linear import LinearModel


def _initial_kernel(v_0):
    return Delta(v_0, event_dim=1)


def _parameter_transform(rho, lamda, s):
    cos_lam = rho * torch.cos(lamda)
    sin_lam = rho * torch.sin(lamda)

    a_top = torch.stack([cos_lam, sin_lam], dim=-1)
    a_bottom = torch.stack([-sin_lam, cos_lam], dim=-1)

    a = torch.stack([a_top, a_bottom], dim=-2)
    new_s = s.unsqueeze(-1).expand(a.shape[:-1])

    return a, torch.zeros_like(new_s), new_s


class CyclicalProcess(LinearModel):
    """
    Implements a cyclical process like `statsmodels`_.

    .. _`statsmodels`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html#statsmodels.tsa.statespace.structural.UnobservedComponents
    """

    def __init__(self, rho: ParameterType, lamda: ParameterType, sigma: ParameterType, x_0: ParameterType = None):
        """
        Internal initializer for :class:`Cyclical`.

        Args:
            rho (ParameterType): see reference.
            lamda (ParameterType): see reference.
            sigma (ParameterType): see reference.
            x_0 (ParameterType): initial values.
        """

        rho, lamda, sigma = broadcast_all(rho, lamda, sigma)

        if x_0 is None:
            x_0 = torch.zeros(2, device=lamda.device)

        distribution = Normal(
            torch.tensor(0.0, device=rho.device), torch.tensor(1.0, device=rho.device)
            ).expand(torch.Size([2])).to_event(1)
        
        super().__init__((rho, lamda, sigma), distribution, _initial_kernel, initial_parameters=(x_0,), parameter_transform=_parameter_transform)

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        new = self.__new__(CyclicalProcess)
        new.__init__(*new_parameters["parameters"], *new_parameters["initial_parameters"])

        return new
