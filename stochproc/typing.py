from numbers import Number
from typing import Union

import torch

ParameterType = Union[Number, torch.Tensor, torch.nn.Parameter]
