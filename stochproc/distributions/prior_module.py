import warnings
from abc import ABC
from collections import OrderedDict
from typing import Iterator, Tuple, Dict, Union

import torch
from torch.nn import Module, ModuleDict, ParameterDict

from .parameter import PriorBoundParameter
from .prior import Prior
from ..container import BufferDict
from ..typing import NamedParameter, ParameterType


class UpdateParametersMixin(ABC):
    def sample_params_(self, shape: torch.Size = torch.Size([])):
        """
        Samples the parameters of the model in place.

        Args:
            shape: shape of the parameters to use when sampling.
        """

        raise NotImplementedError()


def _recurse(obj: "_HasPriorsModule", name_):
    name_split = name_.split(".")

    if len(name_split) == 2:
        return obj.prior_dict[name_split[-1]]

    return _recurse(obj._modules[name_split[0]], ".".join(name_split[1:]))


class _HasPriorsModule(Module, UpdateParametersMixin, ABC):
    """
    Abstract base class that allows registering priors.
    """

    def __init__(self):
        """
        Initializes the :class:`_HasPriorsModule` class.
        """

        super().__init__()

        self.prior_dict = ModuleDict()
        self._parameter_order = list()

        # Bug for ``torch.nn.ParameterDict`` as ``__setitem__`` is disallowed, but ``Module`` initializes training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.parameter_dict = ParameterDict()
            self.buffer_dict = BufferDict()

    def _register_parameter_or_prior(self, name: str, p: ParameterType):
        """
        Helper method for registering either a:
            - :class:`stochproc.distributions.Prior`
            - :class:`torch.nn.Parameter`
            - :class:`torch.Tensor`

        Args:
            name: The name to use for the object.
            p: The object to register.
        """

        if not isinstance(p, NamedParameter):
            p = NamedParameter(name, p)

        if p.prior is not None:
            self.register_prior(p.name, p.prior, p.value)
        elif isinstance(p.value, torch.nn.Parameter):
            self.parameter_dict[p.name] = p.value
        else:
            self.buffer_dict[p.name] = p.value

        self._parameter_order.append(p.name)

    def parameters_and_buffers(self) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter, PriorBoundParameter]]:
        """
        Returns the union of the parameters and buffers of the module in order of registration.
        """

        # TODO: Do this better
        res = OrderedDict()
        for name in self._parameter_order:
            res[name] = self.buffer_dict[name] if name in self.buffer_dict else self.parameter_dict[name]

        return res

    def register_prior(self, name: str, prior: Prior, parameter=None):
        """
        Registers a :class:`stochproc.distributions.Prior` object together with a
        :class:`stochproc.distributions.PriorBoundParameter`.

        Args:
            name: The name to use for the prior.
            prior: The prior of the parameter.
            parameter: Override parameter.
        """

        self.prior_dict[name] = prior
        self.parameter_dict[name] = (
            parameter if parameter is not None else PriorBoundParameter(prior().sample(), requires_grad=False)
        )

    def parameters_and_priors(self) -> Iterator[Tuple[PriorBoundParameter, Prior]]:
        """
        Returns the priors and parameters of the module as an iterable of tuples, i.e::

            (parameter_0, prior_parameter_0), ..., (parameter_n, prior_parameter_n)
        """

        for name, parameter in self.named_parameters():
            yield parameter, _recurse(self, name)

    def sample_params_(self, shape: torch.Size = torch.Size([])):
        for param, prior in self.parameters_and_priors():
            param.sample_(prior, shape)

        return self
