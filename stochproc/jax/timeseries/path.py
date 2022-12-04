import jax.numpy as jnp

from ...path import _StochasticProcessPath, ArrayPath


class StochasticProcessPath(_StochasticProcessPath[jnp.ndarray]):
    def get_path(self) -> ArrayPath[jnp.ndarray]:
        time_indexes = jnp.stack([c.time_index for c in self.path])
        x = jnp.stack([c.value for c in self.path])

        return ArrayPath(time_indexes, x)
