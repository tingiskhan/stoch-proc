from numbers import Number
from typing import Sequence, Union
import torch


SizeLike = Sequence[int]
ParameterType = Union[Number, torch.Tensor, torch.nn.Parameter]
