from torch.distributions import Distribution, AffineTransform, TransformedDistribution
import torch
from typing import Tuple
from .stochastic_process import StructuralStochasticProcess
from ..distributions import DistributionModule
from .typing import MeanScaleFun
from .state import TimeseriesState


def _define_transdist(
    loc: torch.Tensor, scale: torch.Tensor, n_dim: int, dist: Distribution
) -> TransformedDistribution:
    """
    Helper method for defining an affine transition density given the location and scale.

    Args:
        loc: The location of the distribution.
        scale: The scale of the distribution.
        n_dim: The dimension of space of the distribution.
        dist: The base distribution to apply the location-scale transform..

    Returns:
        The resulting affine transformed distribution.
    """

    loc, scale = torch.broadcast_tensors(loc, scale)
    batch_shape = loc.shape[:loc.dim() - n_dim]

    return TransformedDistribution(
        dist.expand(batch_shape), AffineTransform(loc, scale, event_dim=n_dim), validate_args=False
    )


class AffineProcess(StructuralStochasticProcess):
    """
    Class for defining stochastic processes of affine nature, i.e. where we can express the next state :math:`X_{t+1}`
    given the previous state :math:`X_t` as:
        .. math::
            X_{t+1} = f(X_t, \\theta) + g(X_t, \\theta) \\cdot W_{t+1},

    where :math:`\\theta` denotes the parameter set governing the functions :math:`f` and :math:`g`, and :math:`W_t`
    denotes random variable with arbitrary density (from which we can sample).

    Example:
        One example of an affine stochastic process is the AR(1) process. We define it by:
            >>> from stochproc.timeseries import AffineProcess, NamedParameter
            >>> from stochproc.distributions import DistributionModule
            >>> from torch.distributions import Normal, TransformedDistribution, AffineTransform
            >>>
            >>> def mean_scale(x, alpha, beta, sigma):
            >>>     return alpha + beta * x.values, sigma
            >>>
            >>> def init_transform(model, normal_dist):
            >>>     alpha, beta, sigma = model.functional_parameters()
            >>>     return TransformedDistribution(normal_dist, AffineTransform(alpha, sigma / (1 - beta ** 2)).sqrt())
            >>>
            >>> parameters = (
            >>>     NamedParameter("alpha", 0.0),
            >>>     NamedParameter("beta", 0.99),
            >>>     NamedParameter("sigma", 0.05),
            >>> )
            >>>
            >>> initial_dist = increment_dist = DistributionModule(Normal, loc=0.0, scale=1.0)
            >>> ar_1 = AffineProcess(mean_scale, parameters, initial_dist, increment_dist, initial_transform=init_transform)
            >>>
            >>> samples = ar_1.sample_path(1_000)
    """

    def __init__(
        self,
        mean_scale: MeanScaleFun,
        parameters,
        initial_dist,
        increment_dist: DistributionModule,
        **kwargs
    ):
        """
        Initializes the ``AffineProcess`` class.

        Args:
            mean_scale: Function constructing the mean and scale.
            parameters: See base.
            initial_dist: See base.
            increment_dist: Corresponds to the distribution that we location-scale transform.
        """

        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)

        self.mean_scale_fun = mean_scale
        self.increment_dist = increment_dist

    def build_density(self, x):
        loc, scale = self.mean_scale(x)

        return _define_transdist(loc, scale, self.n_dim, self.increment_dist())

    def mean_scale(self, x: TimeseriesState, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process evaluated at ``x`` and ``.functional_parameters()`` or ``parameters``.

        Args:
            x: The previous state of the process.
            parameters: Whether to override the current parameters of the model, otherwise uses
                ``.functional_parameters()``.

        Returns:
            Returns the tuple ``(mean, scale)`` given by evaluating ``(f(x, *parameters), g(x, *parameters))``.
        """

        return self.mean_scale_fun(x, *(parameters or self.functional_parameters()))

    def propagate_conditional(
            self, x: TimeseriesState, u: torch.Tensor, parameters=None, time_increment=1.0
    ) -> TimeseriesState:
        super(AffineProcess, self).propagate_conditional(x, u, parameters, time_increment)

        for _ in range(self.num_steps):
            loc, scale = self.mean_scale(x, parameters=parameters)
            x = x.propagate_from(values=loc + scale * u, time_increment=time_increment)

        return x
