__version__ = "0.0.10"

from torch.distributions import Distribution

Distribution.set_default_validate_args(False)

from . import distributions
from . import timeseries
