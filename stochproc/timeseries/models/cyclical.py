import torch
from pyro.distributions import Delta, Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..affine import AffineProcess


def _initial_kernel(v_0):
    return Delta(v_0, event_dim=1)


def _mean_scale(x, rho, lamda, sigma):
    lt = lamda * x.time_index
    
    top = torch.stack((lt.cos(), lamda.sin()), dim=-1)
    bottom = torch.stack((-lt.sin(), lamda.cos()), dim=-1)
    mat = torch.stack((top, bottom), dim=-2)
    
    c = mat @ x.value.unsqueeze(-1)

    return rho * c.squeeze(-1), sigma.unsqueeze(-1)


class Cyclical(AffineProcess):
    """
    Implements a cyclical process like `statsmodels`_.

    .. _`statsmodels`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html#statsmodels.tsa.statespace.structural.UnobservedComponents
    """

    def __init__(self, rho: ParameterType, lamda: ParameterType, sigma: ParameterType, v_0: ParameterType = torch.zeros(2)):
        """
        Internal initializer for :class:`Cyclical`.

        Args:
            rho (ParameterType): see reference.
            lamda (ParameterType): see reference.
            sigma (ParameterType): see reference.
        """

        rho, lamda, sigma = broadcast_all(rho, lamda, sigma)
        distribution = Normal(
            torch.tensor(0.0, device=rho.device), torch.tensor(1.0, device=rho.device)
            ).expand(torch.Size([2])).to_event(1)
        
        super().__init__(_mean_scale, (rho, lamda, sigma), distribution, _initial_kernel, initial_parameters=(v_0,))


