from abc import ABC
from typing import TypeVar, Generic

from .typing import ShapeLike, TArray, TDistribution
from .state import TimeseriesState
from .path import StochasticProcessPath



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

    def __init__(self, event_shape: ShapeLike) -> None:
        """
        Internal initializer for :class:`StructuralStochasticProcess`.

        Args:
            event_shape (ShapeLike): event shape
        """

        super().__init__()
        self.event_shape = event_shape

    @property
    def n_dim(self) -> int:
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc.
        """

        return len(self.event_shape)

    @property
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it returns the number of elements in the vector or matrix.
        """

        return self.event_shape.numel()

    def initial_distribution(self) -> TDistribution:
        """
        Returns the initial distribution and any re-parameterization given by ``._init_transform``.
        """

        raise NotImplementedError()

    def initial_state(self, *args, shape: ShapeLike = ()) -> TimeseriesState[TArray]:
        """
        Samples the initial state.

        Args:
            shape (ShapeLike): batch shape to sample.

        Returns:
            TimeseriesState[TArray]: initial state of the process.
        """

        raise NotImplementedError()

    def build_distribution(self, x: TimeseriesState[TArray]) -> TDistribution:        
        r"""
        Method to be overridden by derived classes. Defines how to construct the transition density to :math:`X_{t+1}`
        given the state at :math:`t`, i.e. this method corresponds to building the density:
            .. math::
                x_{t+1} \sim p \right ( \cdot \mid \{ x_j \}_{j \leq t} \left ).

        Args:
            x: previous state of the process.

        Returns:
            TDistribution: Returns the density of the state at :math:`t+1`.
        """

        raise NotImplementedError()

    def sample_states(self, steps: int, *args, shape: ShapeLike = (), x0: TimeseriesState[TArray] = None) -> StochasticProcessPath:
        r"""
        Samples a trajectory from the stochastic process, i.e. samples the collection :math:`\{ X_j \}^T_{j = 0}`,
        where :math:`T` corresponds to ``steps``.

        Args:
            steps: number of steps to sample.
            samples: batch shape to sample.
            x0: initial sample to use, if ``None``, samples one.

        Returns:
            StochasticProcessPath: Returns a path sampled from the stochastic process.
        """

        raise NotImplementedError()
