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
