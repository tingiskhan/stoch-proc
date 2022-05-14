from numbers import Number
from typing import Union
import torch
from .distributions import Prior


_ParameterType = Union[Number, torch.Tensor, torch.nn.Parameter, Prior]


class NamedParameter(object):
    """
    Defines a named parameter type.
    """

    def __init__(self, name: str, value: _ParameterType):
        """
        Initializes the ``NamedParameter`` class.

        Args:
            name: The name of the parameter.
            value: The value of the parameter
        """

        self.name = name
        self.value = torch.tensor(value) if isinstance(value, Number) else value


ParameterType = Union[_ParameterType, NamedParameter]
