from stochproc.jax import timeseries as ts
from numpyro.distributions import Normal
from jax.random import PRNGKey, split
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

        a = jnp.ones(shape)
        b = jnp.ones(shape)
        proc = ts.StructuralStochasticProcess(prop, (a, b), init_dist)

        assert proc.event_shape == ()

        key = PRNGKey(0)
        init_state = proc.initial_state(key, shape=shape)

        assert init_state.value.shape == shape
        dist = proc.build_distribution(init_state)

        assert isinstance(dist, Normal) and (dist.batch_shape == shape) and (dist.event_shape == proc.event_shape)

        new_state = proc.propagate_state(init_state, key)

        assert isinstance(new_state, init_state.__class__) and (new_state.value == dist.sample(key)).all() and (new_state.value.shape == shape)
