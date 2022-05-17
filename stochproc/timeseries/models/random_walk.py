from torch.distributions import Normal
from ..linear import LinearModel
from ...distributions import DistributionModule
from ...typing import ParameterType
from ...utils import enforce_named_parameter


class RandomWalk(LinearModel):
    r"""
    Defines a one-dimensional Gaussian random walk process, i.e. in which the dynamics are given by
        .. math::
            X_{t+1} \sim \mathcal{N}(X_t, \sigma), \newline
            X_0 \sim \mathcal{N}(x_0, \sigma_0),

    where :math:`x_0` is the initial mean, defaulting to zero.
    """

    def __init__(self, scale: ParameterType, initial_mean: ParameterType = 0.0, **kwargs):
        """
        Initializes the :class:`RandomWalk` model.

        Args:
            scale: :math:`\\sigma` in class doc.
            initial_mean: parameter specifying the mean of the initial distribution. Defaults to 0.
            kwargs: see base.
        """

        a, scale, initial_mean = enforce_named_parameter(a=1.0, scale=scale, loc=initial_mean)

        initial_dist = DistributionModule(Normal, loc=initial_mean, scale=scale)
        inc_dist = DistributionModule(Normal, loc=0.0, scale=1.0)

        super().__init__(a, scale, increment_dist=inc_dist, initial_dist=initial_dist, **kwargs)
