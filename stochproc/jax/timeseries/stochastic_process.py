import jax.numpy as jnp
from numpyro.distributions import Distribution
from jax.random import KeyArray, split
import jax
from jax.tree_util import register_pytree_node_class

from .state import TimeseriesState
from .path import StochasticProcessPath
from ...stochastic_process import _StructuralStochasticProcess
from ...typing import ShapeLike


@register_pytree_node_class
class StructuralStochasticProcess(_StructuralStochasticProcess[Distribution, jnp.ndarray]):
    """
    Sub class of :class:`_StructuralStochasticProcess`.
    """

    def __init__(self, kernel, parameters, initial_distribution: Distribution):
        """
        Internal initializer for :class:`StructuralStochasticProcess`.

        Args:
            kernel: see base.
            parameters: see base.
            initial_distribution (Distribution): initial distribution.
        """

        super().__init__(kernel, parameters, initial_distribution.event_shape)
        self._initial_distribution = initial_distribution

    def tree_flatten(self):
        children = (self.parameters,)
        aux_data = (self.kernel, self.initial_distribution())

        return (children, aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kernel, init_dist = aux_data        
        return cls(kernel, children[0], init_dist)
    
    def initial_distribution(self):
        return self._initial_distribution

    def initial_state(self, key: KeyArray, shape: ShapeLike = ()):
        initial_dist = self.initial_distribution()
        x_0 = initial_dist.sample(key, shape)

        return TimeseriesState(jnp.array(0, dtype=jnp.int16), x_0, event_shape=self.event_shape)

    @jax.jit
    def propagate_state(self, x: TimeseriesState, key: KeyArray) -> TimeseriesState:
        density = self.build_distribution(x)

        return x.propagate_from(density.sample(key), 1)
    
    def sample_states(self, steps, key: KeyArray, shape=(), x_0: TimeseriesState = None) -> StochasticProcessPath:
        state = x_0 if x_0 else self.initial_state(key, shape)
        states = (state,)

        for __ in range(steps):
            _, key = split(key)
            state = self.propagate_state(state, key)
            states += (state,)

        return StochasticProcessPath(states)
