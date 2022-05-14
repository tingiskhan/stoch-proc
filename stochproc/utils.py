import torch
from torch.distributions import utils
from functools import wraps
from typing import Dict, List
from .distributions.base import _DistributionModule
from .typing import ParameterType, NamedParameter


def concat(*x: torch.Tensor) -> torch.Tensor:
    """
    Given an iterable of tensors, broadcast them to the same shape and stack along the last dimension.

    Args:
        x: The iterable of tensors to stack.
    """

    if isinstance(x, torch.Tensor):
        return x

    return torch.stack(torch.broadcast_tensors(*x), dim=-1)


def construct_diag_from_flat(x: torch.Tensor, base_dim: int) -> torch.Tensor:
    """
    Constructs a diagonal matrix based on ``x``. Solution found `here`_:

    Args:
        x: The diagonal of the matrix.
        base_dim: The dimension of ``x``.

    Example:
        If ``x`` is of shape ``(100, 20)`` and the dimension is 0, then we get a tensor of shape ``(100, 20, 1)``.

    .. _here: https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p
    """

    if base_dim == 0:
        return x.unsqueeze(-1).unsqueeze(-1)

    if base_dim == 1 and x.shape[-1] < 2:
        return x.unsqueeze(-1)

    return x.unsqueeze(-1) * torch.eye(x.shape[-1], device=x.device)


def broadcast_all(*values):
    """
    Wrapper around ``torch.distributions.utils.broadcast_all`` for setting the same shape of tensors while ignoring
    any objects inheriting from ``_DistributionModule``.

    Args:
        values: Iterable of tensors.
    """

    broadcast_tensors = utils.broadcast_all(*(v for v in values if not issubclass(v.__class__, _DistributionModule)))

    res = tuple()
    torch_index = 0
    for i, v in enumerate(values):
        is_dist_subclass = issubclass(v.__class__, _DistributionModule)
        res += (values[i] if is_dist_subclass else broadcast_tensors[torch_index],)

        torch_index += int(not is_dist_subclass)

    return res


def is_documented_by(original):
    """
    Wrapper for function for copying doc strings of functions. See `this`_ reference.

    Args:
        original: The original function to copy the docs from.

    .. _this: https://softwareengineering.stackexchange.com/questions/386755/sharing-docstrings-between-similar-functions
    """

    @wraps(original)
    def wrapper(target):
        target.__doc__ = original.__doc__

        return target

    return wrapper


def enforce_named_parameter(**kwargs: ParameterType) -> List[ParameterType]:
    """
    Enforces parameter types into ``NamedParameters``.

    Args:
        kwargs: Key worded arguments.
    """

    res = list()
    for k, v in kwargs.items():
        if not isinstance(v, NamedParameter):
            res.append(NamedParameter(k, v))
        else:
            # TODO: Perhaps warn?
            res.append(NamedParameter(k, v.value))

    return res
