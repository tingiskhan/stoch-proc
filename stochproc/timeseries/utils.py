from typing import Tuple
import torch

from ..typing import ParameterType


def coerce_tensors(*x: ParameterType) -> Tuple[torch.Tensor, ...]:
    """
    Coerces objects to be tensors and verifies that all are on the same device.

    Returns:
        Tuple[torch.Tensor, ...]: Input parameters coerced to tensors.
    """

    tensors = tuple(p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in x)

    assert all(t.device == tensors[0].device for t in tensors), "All tensors do not have the same device!"

    return tensors


def lazy_property(property_name: str):
    """
    Helper method for evaluating lazy expressions.
    Args:
        property_name (str): name of the property to set on the dictionary.
    """

    def wrapper_outer(f):
        def wrapper_inner(self):
            prop = f(self)

            if callable(prop):
                # TODO: Perhaps verify that self is a dictionary...
                prop = self[property_name] = prop()
                        
            return prop

        return property(wrapper_inner, doc=f.__doc__)

    return wrapper_outer