from torch.distributions import Distribution
import torch
from typing import Type, Callable, Sequence, Union, Dict
from numbers import Number

HyperParameter = Union[torch.Tensor, Number]
DistributionOrBuilder = Union[Type[Distribution], Callable[[Dict[str, HyperParameter]], Distribution]]
