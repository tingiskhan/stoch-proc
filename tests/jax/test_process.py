from stochproc.jax import timeseries as ts
from numpyro.distributions import Normal
from jax.random import PRNGKey
import jax.numpy as jnp
import jax
import pytest as pt


@jax.jit
def prop(x, a, b):
    return Normal(a * x.value, b)


class TestTimeseries(object):    
    @pt.mark.parametrize("shape", [(), (10,), (10, 10), (2_000, 500)])
    def test_initialize_and_sample(self, shape):
        init_dist = Normal()
        proc = ts.StructuralStochasticProcess(init_dist)

        assert proc.event_shape == ()

        key = PRNGKey(0)
        init_state = proc.initial_state(key, shape=shape)

        assert init_state.value.shape == shape

        a = jnp.ones(shape)
        b = jnp.ones(shape)

        j = prop(init_state, a, b)
        print()
