import jax.numpy as jnp
from numpyro.distributions import Distribution
from jax.random import PRNGKey

from ...stochastic_process import StructuralStochasticProcess
from ...typing import ShapeLike
from ...state import TimeseriesState
from ...path import ProcessPath


class JaxProcess(StructuralStochasticProcess[Distribution, jnp.ndarray]):
    """
    Base class for JAX based stochastic processes.
    """

    def __init__(self, initial_distribution: Distribution):
        """
        Internal initializer for :class:`JaxProcess`.

        Args:
            initial_distribution (Distribution): initial distribution.
        """

        super().__init__(initial_distribution.event_shape)
        self._initial_distribution = initial_distribution
    
    def initial_distribution(self):
        return self._initial_distribution
