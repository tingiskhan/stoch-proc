import warnings
from torch.nn import Module, ModuleDict, ParameterDict
from abc import ABC
import torch
from collections import OrderedDict
from typing import Iterator, Tuple, Dict, Union
from .parameter import PriorBoundParameter
from ..container import BufferDict
from .prior import Prior
from ..typing import NamedParameter, _ParameterType


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
            >>> from pyfilter import timeseries as ts, distributions as dists
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

    def _register_parameter_or_prior(self, name: str, p: Union[_ParameterType, NamedParameter]):
        """
        Helper method for registering either a:
            - ``pyfilter.distributions.Prior``
            - ``torch.nn.Parameter``
            - ``torch.Tensor``

        Args:
            name: The name to use for the object.
            p: The object to register.
        """

        if isinstance(p, NamedParameter):
            name = p.name
            p = p.value

        if isinstance(p, Prior):
            self.register_prior(name, p)
        elif isinstance(p, torch.nn.Parameter):
            self.parameter_dict[name] = p
        else:
            self.buffer_dict[name] = p if (isinstance(p, torch.Tensor) or p is None) else torch.tensor(p)

        self._parameter_order.append(name)

    def parameters_and_buffers(self) -> Dict[str, Union[torch.Tensor, torch.nn.Parameter]]:
        """
        Returns the union of the parameters and buffers of the module.
        """

        # TODO: Do this better
        res = OrderedDict()
        for name in self._parameter_order:
            res[name] = self.buffer_dict[name] if name in self.buffer_dict else self.parameter_dict[name]

        return res

    def register_prior(self, name: str, prior: Prior):
        """
        Registers a ``pyfilter.distributions.Prior`` object together with a ``pyfilter.PriorBoundParameter`` on self. If
        the same prior object already exists on the object, it's assumed that the corresponding parameter should be the
        same object.

        Args:
            name: The name to use for the prior.
            prior: The prior of the parameter.
        """

        parameter = next(
            (prm for (prm, p) in self.parameters_and_priors() if p is prior),
            PriorBoundParameter(prior().sample(), requires_grad=False)
        )

        self.prior_dict[name] = prior
        self.parameter_dict[name] = parameter

    def parameters_and_priors(self) -> Iterator[Tuple[PriorBoundParameter, Prior]]:
        """
        Returns the priors and parameters of the module as an iterable of tuples, i.e::
            [(parameter_0, prior_parameter_0), ..., (parameter_n, prior_parameter_n)]
        """

        prior_memo = set()

        for name, prior in self.prior_dict.items():
            prior_memo.add(prior)
            parameter = self.parameter_dict[name]

            yield parameter, prior

        for module in filter(lambda u: isinstance(u, _HasPriorsModule), self.children()):
            for parameter, prior in module.parameters_and_priors():
                if prior in prior_memo:
                    continue

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
