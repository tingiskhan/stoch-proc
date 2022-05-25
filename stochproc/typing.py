from numbers import Number
from typing import Union
import torch
from .distributions import Prior


_ParameterType = Union[Number, torch.Tensor, torch.nn.Parameter, Prior]


class NamedParameter(object):
    """
    Defines a named parameter.
    """

    def __init__(self, name: str, value: _ParameterType):
        """
        Initializes the :class:`NamedParameter` class.

        Args:
            name: name of the parameter.
            value: value of the parameter
        """

        from .distributions import PriorBoundParameter

        self.name = name

        is_prior = isinstance(value, Prior)
        self.prior = value if is_prior else None
        if is_prior:
            self.value = PriorBoundParameter(self.prior().sample(), requires_grad=False)
        else:
            self.value = value if isinstance(value, torch.Tensor) else torch.tensor(value)


ParameterType = Union[_ParameterType, NamedParameter]
