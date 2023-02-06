# Changelog

## Versions

### v0.2.6
 - Adds the `HiddenMarkovModel`.

### v0.2.5
 - Use `Delta` distribution for some models.
 - Improved handling of joint state by skipping creating from scratch all the time.
 - Adds `expand` method for creating batched sampling.

### v0.2.4
 - Adds `CyclicalProcess` and "fixes" `HarmonicProcess`

### v0.2.3
 - Bug fixes for `DoubleExponential` and `SelfExcitingProcess`.

### v0.2.1
 - Uses tuple instead of list for parameters

### v0.2.0
 - Improves API regarding overriding parameters

### v0.1.2
 - Improves `LinearModel`
 - Adds `LinearStateModel`
 - Minor test improvements
 - Adds `HarmonicProcess`

### v0.1.0
 - Simplifies backend by removing all dependencies to `torch.nn.Module` to facilitate supporting JAX.
 - Migrates to pyproject.toml file instead.

### v0.0.24
 - Adds support for scaling sub process in `SmoothLinearTrend`

### v0.0.23
 - Adds `LowerCholeskyJointStochasticProcess`
 - Adds `LowerCholeskyHierarchicalProcess`
 - Adds primitive `joint_processes` to facilitate joining processes

### v0.0.22
 - Adds the bivariate Trending OU process.
 - Adds `JointStochasticProcess` for joint processes of arbitrary dynamics.
 - `JointDistribution` now coerces all of the distributions to have the same `batch_shape`

### v0.0.21
 - Adds the Trending OU process

### v0.0.14
 - Adds the `batch_shape` property to `TimeseriesState`.
 - Renames `event_dim` property of `TimeseriesState` to `event_shape`.
 - Adds support for `expand` and `to_event` on `DistributionModule`.

### v0.0.13
 - Adds the `SmoothLinearTrend` model. 

### v0.0.12
 - Minor cleanup and improvement in `diffusion.py` module as we now use the base class' implementation of `mean_scale` 
   and removes useless class.
 - Adds support for defining hierarchical timeseries. Documentation is somewhat lacking.