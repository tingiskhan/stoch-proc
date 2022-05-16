import warnings
from torch.nn import Module, ModuleDict, ParameterDict
from abc import ABC
import torch
from collections import OrderedDict
from typing import Iterator, Tuple, Dict, Union
from .parameter import PriorBoundParameter
from ..container import BufferDict
from .prior import Prior
from ..typing import NamedParameter, ParameterType


class UpdateParametersMixin(ABC):
    def sample_params_(self, shape: torch.Size = torch.Size([])):
        """
        Samples the parameters of the model in place.

        Args:
            shape: The shape of the parameters to use when sampling.
        """

        raise NotImplementedError()

    def concat_parameters(self, constrained=False, flatten=True) -> torch.Tensor:
        """
        Concatenates the parameters into one tensor.

        Args:
            constrained: Optional parameter specifying whether to concatenate the original parameters, or bijected.
            flatten: Optional parameter specifying whether to flatten the parameters.
        """

        raise NotImplementedError()

    def update_parameters_from_tensor(self, x: torch.Tensor, constrained=False):
        """
        Update the parameters of ``self`` with the last dimension of ``x``.

        Args:
            x: The tensor containing the new parameter values.
            constrained: Optional parameter indicating whether values in ``x`` are considered constrained to the
                parameters' original space.

        Example:
            >>> from stochproc import timeseries as ts, distributions as dists
            >>> from torch.distributions import Normal, Uniform
            >>> import torch
            >>>
            >>> alpha_prior = dists.Prior(Normal, loc=0.0, scale=1.0)
            >>> beta_prior = dists.Prior(Uniform, low=-1.0, high=1.0)
            >>>
            >>> ar = ts.models.AR(alpha_prior, beta_prior, 0.05)
            >>> ar.sample_params_(torch.Size([1]))
            >>>
            >>> new_values = torch.empty(2).normal_()
            >>> ar.update_parameters_from_tensor(new_values, constrained=False)
            >>> assert (new_values == ar.concat_parameters(constrained=False)).all()
        """

        raise NotImplementedError()

    def eval_prior_log_prob(self, constrained=True) -> torch.Tensor:
        """
        Calculates the prior log-likelihood of the current values of the parameters.

        Args:
            constrained: Optional parameter specifying whether to evaluate the original prior, or the bijected prior.
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
        Initializes the ``_HasPriorsModule`` class.
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
            - ``stochproc.distributions.Prior``
            - ``torch.nn.Parameter``
            - ``torch.Tensor``

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

    def parameters_and_buffers(self) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
        """
        Returns the union of the parameters and buffers of the module.
        """

        # TODO: Do this better
        res = OrderedDict()
        for name in self._parameter_order:
            res[name] = self.buffer_dict[name] if name in self.buffer_dict else self.parameter_dict[name]

        return res

    def register_prior(self, name: str, prior: Prior, parameter=None):
        """
        Registers a ``stochproc.distributions.Prior`` object together with a ``pyfilter.PriorBoundParameter`` on self.
        Utilizes the same parameter for same ``NamedParameter``.

        Args:
            name: The name to use for the prior.
            prior: The prior of the parameter.
            parameter: Override parameter.
        """

        self.prior_dict[name] = prior
        self.parameter_dict[name] = parameter if parameter is not None else PriorBoundParameter(prior().sample(), requires_grad=False)

    def parameters_and_priors(self) -> Iterator[Tuple[PriorBoundParameter, Prior]]:
        """
        Returns the priors and parameters of the module as an iterable of tuples, i.e::
            [(parameter_0, prior_parameter_0), ..., (parameter_n, prior_parameter_n)]
        """

        for name, parameter in self.named_parameters():
            prior = _recurse(self, name)

            yield parameter, prior

    def sample_params_(self, shape: torch.Size = torch.Size([])):
        for param, prior in self.parameters_and_priors():
            param.sample_(prior, shape)

        return self

    def eval_prior_log_prob(self, constrained=True) -> torch.Tensor:
        return sum((prior.eval_prior(p, constrained) for p, prior in self.parameters_and_priors()))

    def concat_parameters(self, constrained=False, flatten=True) -> Union[torch.Tensor, None]:
        def _first_dim(p: PriorBoundParameter, prior: Prior):
            return (-1,) if flatten else p.shape[: p.dim() - len(prior.shape)]

        res = tuple(
            (p if constrained else prior.get_unconstrained(p)).view(*_first_dim(p, prior), prior.get_numel(constrained))
            for p, prior in self.parameters_and_priors()
        )

        if not res:
            return None

        return torch.cat(res, dim=-1)

    def update_parameters_from_tensor(self, x: torch.Tensor, constrained=False):
        left_index = 0
        for p, prior in self.parameters_and_priors():
            right_index = left_index + prior.get_numel(constrained=constrained)

            p.update_values_(x[..., left_index:right_index], prior, constrained=constrained)
            left_index = right_index
