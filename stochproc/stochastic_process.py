from abc import ABC
from typing import TypeVar, Generic

from .state import TimeseriesState, TArray


TDistribution = TypeVar("TDistribution")


class StructuralStochasticProcess(Generic[TDistribution, TArray]):
    r"""
    Abstract base class for structural stochastic processes. By "stochastic process" we mean a sequence of random variables,
    :math:`\{X_t\}_{t \in T}`, defined on a common probability space
    :math:`\{ \Omega, \mathcal{F}, \{ \mathcal{F}_t \}`, with joint distribution
        .. math::
            p(x_0, ..., x_t) = p(x_0) \prod^t_{k=1} p(x_k \mid x_{1:k-1}),
    
    we further assume that assume that the conditional distribution
    :math:`p(x_k \mid  x_{1:k-1})` is further parameterized by a collection of parameters :math:`\theta`, s.t.
        .. math::
            p_{\theta}(x_k \mid x_{1:k-1}) = p(x_k \mid x_{1:k-1}, \theta).

    Derived classes should override the ``.build_distribution(...)`` method, which builds the distribution of
    :math:`X_{t+1}` given :math:`\{ X_j \}_{j \leq t}`.
    """

    def build_distribution(self, x: TimeseriesState[TArray]) -> TDistribution:
        r"""
        Method to be overridden by derived classes. Defines how to construct the transition density to :math:`X_{t+1}`
        given the state at :math:`t`, i.e. this method corresponds to building the density:
            .. math::
                x_{t+1} \sim p \right ( \cdot \mid \{ x_j \}_{j \leq t} \left ).

        Args:
            x: previous state of the process.

        Returns:
            Returns the density of the state at :math:`t+1`.
        """

        raise NotImplementedError()
