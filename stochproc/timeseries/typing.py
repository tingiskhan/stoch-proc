from typing import Callable, Tuple

import torch
from torch.distributions import Distribution

from .state import TimeseriesState

MeanScaleFun = Callable[[TimeseriesState, Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, torch.Tensor]]
DiffusionFunction = Callable[[TimeseriesState, float, Tuple[torch.Tensor, ...]], Distribution]
Drift = Callable[[TimeseriesState, Tuple[torch.Tensor, ...]], torch.Tensor]
