from torch.distributions import Normal
import torch
from ..linear import LinearModel
from ...distributions import DistributionModule
from ...typing import ParameterType
from ...utils import enforce_named_parameter


class RandomWalk(LinearModel):
    """
    Defines a Gaussian random walk process, i.e. in which the dynamics are given by
        .. math::
            X_{t+1} \\sim \\mathcal{N}(X_t, \\sigma), \n
            X_0 \\sim \\mathcal{N}(x_0, \\sigma_0),

    where :math:`x_0` is the initial mean, defaulting to zero.
    """

    def __init__(self, scale: ParameterType, initial_mean: ParameterType = 0.0, **kwargs):
        """
        Initializes the ``RandomWalk`` model.

        Args:
            scale: Corresponds to :math:`\\sigma` in class doc.
            initial_mean: Optional parameter specifying the mean of the initial distribution. Defaults to 0.
            initial_scale: Optional parameter specifying the scale of the initial distribution. Defaults to 1.
            kwargs: See base.
        """

        a, scale, initial_mean = enforce_named_parameter(a=1.0, scale=scale, loc=initial_mean)

        reinterpreted_batch_ndims = None if len(scale.value.shape) == 0 else 1
        initial_dist = DistributionModule(
            Normal, loc=initial_mean, scale=scale, reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )

        inc_dist = DistributionModule(
            Normal,
            loc=torch.zeros(initial_mean.value.shape),
            scale=torch.ones(initial_mean.value.shape),
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )

        super().__init__(a, scale, increment_dist=inc_dist, initial_dist=initial_dist, **kwargs)
