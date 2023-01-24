import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from numpyro.distributions import Distribution, TransformedDistribution, transforms

from .backend import Backend


class Jax(Backend[jnp.DeviceArray, Distribution]):
    """
    Implements torch as backend.
    """

    def __init__(self) -> None:
        self.register_states()

    def coerce_arrays(self, *x):
        tensors = tuple(p if isinstance(p, jnp.DeviceArray) else jnp.array(p) for p in x)

        return tensors
    
    def broadcast_arrays(self, *x):
        return jnp.broadcast_arrays(*x)
    
    def affine_transform(self, base, loc, scale, n_dim: int):
        return TransformedDistribution(base, transforms.AffineTransform(loc, scale))

    def register_states(self):
        from stochproc.timeseries.state import TimeseriesState

        def flatten_func(state: TimeseriesState):
            return (state.time_index, state.value), (state.event_shape,)

        def unflatten_func(aux_data, children):
            return TimeseriesState(*children, *aux_data)

        register_pytree_node(TimeseriesState, flatten_func, unflatten_func)
