from stochproc.jax import timeseries as ts
from numpyro.distributions import Normal
from jax.random import PRNGKey


class TestTimeseries(object):    
    def test_initialize(self):
        init_dist = Normal()
        proc = ts.JaxProcess(init_dist)

        assert proc.event_shape == ()

        key = PRNGKey(0)
        init_values = proc.initial_distribution().sample(key)

        assert init_values.shape == ()

