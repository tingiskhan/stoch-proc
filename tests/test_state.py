from stochproc.state import TimeseriesState
import jax.numpy as jnp
import pytest as pt
import torch

JAX_ARRAY = jnp.ones(10)
TORCH_ARRAY = torch.ones(10)

class TestState(object):
    @pt.mark.parametrize("array", [JAX_ARRAY, lambda: JAX_ARRAY])
    def test_jax(self, array):
        state = TimeseriesState(jnp.array(0), array, ())

        assert (state.values == JAX_ARRAY).all()

    @pt.mark.parametrize("array", [TORCH_ARRAY, lambda: TORCH_ARRAY])
    def test_torch(self, array):
        state = TimeseriesState(torch.tensor(0), array, ())

        assert (state.values == TORCH_ARRAY).all()
