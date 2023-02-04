from typing import Callable, Union
from collections import OrderedDict

import torch

from .utils import lazy_property

LazyTensor = Union[torch.Tensor, Callable[[], torch.Tensor]]


class TimeseriesState(dict):
    """
    State object for ``StochasticProcess``.
    """

    def __init__(
        self,
        time_index: Union[int, torch.IntTensor],
        values: LazyTensor,
        event_shape: torch.Size,
    ):
        """
        Internal initializer for :class:`TimeseriesState`.

        Args:
            time_index: time index of the state.
            values: values of the state. Can be a lazy evaluated tensor as well.
            event_shape: event dimension.
            exogenous: whether to include any exogenous data.
        """

        super().__init__()

        self["_time_index"] = (time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)).int()
        self["_value"] = values
        self["_event_shape"] = event_shape

    @property
    def time_index(self) -> torch.IntTensor:
        return self["_time_index"]

    @lazy_property("_value")
    def value(self) -> torch.Tensor:
        return self["_value"]

    @property
    def event_shape(self) -> torch.Size:
        return self["_event_shape"]

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the batch shape.
        """

        tot_dim = self.value.dim()
        event_dim = len(self.event_shape)

        return self.value.shape[: tot_dim - event_dim]

    def copy(self, values: LazyTensor) -> "TimeseriesState":
        """
        Copies self with specified ``values``, but with ``time_index`` of current instance.

        Args:
            values: see ``__init__``.
        """

        prop = self.propagate_from(values=values, time_increment=0)

        return prop

    def propagate_from(self, values: LazyTensor, time_increment: int = 1):
        """
        Returns a new instance of :class:`TimeseriesState` with `values`` and ``time_index`` given by
        ``.time_index + time_increment``.

        Args:
            values: see ``__init__``.
            time_increment: how much to increase ``.time_index`` with for new state.
        """

        return TimeseriesState(time_index=self.time_index + time_increment, values=values, event_shape=self.event_shape)

    def __repr__(self):
        return f"TimeseriesState at t={self.time_index} containing: {self.value.__repr__()}"


class JointState(TimeseriesState):
    """
    Implements a joint state for joint timeseries.
    """

    _ENFORCE_SAME_TIME_INDEX = True

    def __init__(self, **sub_states: TimeseriesState):
        """
        Internal initializer for :class:`JointState`.

        Args:
            sub_states: The sub states.
            enforce_same_time_index
        """

        time_index = tuple(sub_states.values())[0].time_index
        event_dim = torch.Size([sum(ss.event_shape.numel() for ss in sub_states.values())])
        super().__init__(time_index=time_index, event_shape=event_dim, values=None)

        self._sub_states_order = tuple(sub_states.keys())
        for name, sub_state in sub_states.items():
            if self._ENFORCE_SAME_TIME_INDEX:
                assert (sub_state.time_index == time_index).all()

            self[name] = sub_state

    @property
    def value(self) -> torch.Tensor:
        res = tuple()

        for sub_state_name in self._sub_states_order:
            sub_state: TimeseriesState = self[sub_state_name]
            res += (sub_state.value if any(sub_state.event_shape) else sub_state.value.unsqueeze(-1),)

        return torch.cat(res, dim=-1)

    def propagate_from(self, values: LazyTensor, time_increment=1.0):
        # NB: This is a hard assumption that the values are in the correct order...
        last_ind = 0

        if callable(values):
            values = values()

        result = OrderedDict([])
        for sub_state_name in self._sub_states_order:
            sub_state: TimeseriesState = self[sub_state_name]

            dimension = sub_state.event_shape.numel()
            sub_values = values[..., last_ind : last_ind + dimension].squeeze(-1)

            result[sub_state_name] = sub_state.propagate_from(sub_values, time_increment=time_increment)
            last_ind += dimension

        new = self.__new__(type(self))
        super(JointState, new).__init__(self.time_index + time_increment, None, self.event_shape)
        new._sub_states_order = self._sub_states_order
        new.update(result)

        return new


class StateSpaceModelState(JointState):
    """
    Implements a joint state for joint timeseries.
    """

    _ENFORCE_SAME_TIME_INDEX = False
