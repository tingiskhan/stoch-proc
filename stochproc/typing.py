from numbers import Number
from typing import Union, NamedTuple
import torch
from .distributions import Prior


_ParameterType = Union[Number, torch.Tensor, torch.nn.Parameter, Prior]
NamedParameter = NamedTuple("NamedParameter", name=str, value=_ParameterType)
ParameterType = Union[_ParameterType, NamedParameter]
