import torch
from torch.distributions import Normal
from .ssm import StateSpaceModel
from .observable import LinearObservations
from ..distributions import DistributionModule
from ..typing import ParameterType
from ..utils import enforce_named_parameter


class LinearSSM(StateSpaceModel):
    r"""
    Defines a state space model where the observation dynamics are given by a linear combination of the latent states
        .. math::
            Y_t = A \cdot X_t + \sigma W_t,

    where :math:`A` is a matrix of size ``(dimension of observation space, dimension of latent space)``, :math:`W_t` is
    a random variable with arbitrary density, and :math:`\sigma` is a scaling parameter.
    """

    def __init__(self, hidden, a: ParameterType, scale: ParameterType, base_dist: DistributionModule):
        """
        Initializes the :class:`LinearObservations` class.

        Args:
            hidden: hidden process.
            a: matrix :math:`A`.
            scale: scale of the.
            base_dist: base distribution.
        """

        observable = LinearObservations(a, scale, base_dist)
        super().__init__(hidden, observable)


class LinearGaussianObservations(LinearSSM):
    """
    Same as :class:`LinearObservations` but where the distribution :math:`W_t` is given by a Gaussian distribution with
    zero mean and unit variance.
    """

    def __init__(self, hidden, a=1.0, scale=1.0):
        """
        Initializes the :class:`LinearGaussianObservations` class.

        Args:
            hidden: see base.
            a: see base.
            scale: see base.
        """

        a, scale = enforce_named_parameter(a=a, scale=scale)

        if len(a.value.shape) < 2:
            dist = DistributionModule(Normal, loc=0.0, scale=1.0)
        else:
            dim = a.value.shape[-2]
            dist = DistributionModule(Normal, loc=torch.zeros(dim), scale=torch.ones(dim), reinterpreted_batch_ndims=1)

        super().__init__(hidden, a, scale, dist)
