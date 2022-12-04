from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

from ...state import _TimeseriesState


@register_pytree_node_class
class TimeseriesState(_TimeseriesState[jnp.ndarray]):
    def tree_flatten(self):
        children = (self.time_index, self.value)
        aux_data = (self.event_shape,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)