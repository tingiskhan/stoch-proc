import jax.numpy as jnp
import pytest as pt
import torch
from stochproc.state import _TimeseriesState


JAX_ARRAY = jnp.ones(10)
TORCH_ARRAY = torch.ones(10)


class TestState(object): 
    @pt.mark.parametrize("array", [JAX_ARRAY, lambda: JAX_ARRAY])
    def test_jax(self, array):
        state = _TimeseriesState(jnp.array(0), array, ())

        assert (state.value == JAX_ARRAY).all()

    @pt.mark.parametrize("array", [TORCH_ARRAY, lambda: TORCH_ARRAY])
    def test_torch(self, array):
        state = _TimeseriesState(torch.tensor(0), array, ())

        assert (state.value == TORCH_ARRAY).all()
