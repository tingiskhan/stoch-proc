# Changelog

## Versions

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