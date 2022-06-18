from typing import Union, Callable

import torch

LazyTensor = Union[torch.Tensor, Callable[[], torch.Tensor]]


class TimeseriesState(dict):
    """
    State object for ``StochasticProcess``.
    """

    def __init__(
        self,
        time_index: Union[int, torch.IntTensor],
        values: LazyTensor,
        event_dim: torch.Size,
        exogenous: torch.Tensor = None,
    ):
        """
        Initializes the :class:`TimeseriesState` class.

        Args:
            time_index: time index of the state.
            values: values of the state. Can be a lazy evaluated tensor as well.
            event_dim: event dimension.
            exogenous: whether to include any exogenous data.
        """

        super().__init__()

        self.time_index: torch.IntTensor = (
            time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)
        ).int()
        self.exogenous: torch.Tensor = exogenous
        self.event_dim = event_dim

        self._values = values

    @property
    def values(self) -> torch.Tensor:
        """
        The values of the state.
        """

        if callable(self._values):
            self._values = self._values()

        return self._values

    @values.setter
    def values(self, x):
        self._values = x

    def copy(self, values: LazyTensor) -> "TimeseriesState":
        """
        Copies self with specified ``values``, but with ``time_index`` of current instance.

        Args:
            values: see ``__init__``.
        """

        prop = self.propagate_from(values=values, time_increment=0)
        prop.exogenous = self.exogenous

        return prop

    def propagate_from(self, values: LazyTensor, time_increment: int = 1):
        """
        Returns a new instance of :class:`TimeseriesState` with `values`` and ``time_index`` given by
        ``.time_index + time_increment``.

        Args:
            values: see ``__init__``.
            time_increment: how much to increase ``.time_index`` with for new state.
        """

        return TimeseriesState(time_index=self.time_index + time_increment, values=values, event_dim=self.event_dim)

    def add_exog(self, x: torch.Tensor):
        """
        Adds an exogenous variable to the state.

        Args:
            x: exogenous variable.
        """

        self.exogenous = x

    def __repr__(self):
        return f"TimeseriesState at t={self.time_index} containing: {self.values.__repr__()}"


class JointState(TimeseriesState):
    """
    Implements a joint state for joint timeseries.
    """

    _ENFORCE_SAME_TIME_INDEX = True

    def __init__(self, **sub_states: TimeseriesState):
        """
        Initializes the :class:`JointState` class.

        Args:
            sub_states: The sub states.
            enforce_same_time_index
        """

        time_index = tuple(sub_states.values())[0].time_index
        event_dim = torch.Size([sum(ss.event_dim[0] if any(ss.event_dim) else 1 for ss in sub_states.values())])
        super(JointState, self).__init__(time_index=time_index, event_dim=event_dim, values=None)

        self._sub_states_order = sub_states.keys()
        for name, sub_state in sub_states.items():
            if self._ENFORCE_SAME_TIME_INDEX:
                assert (sub_state.time_index == time_index).all()

            self[name] = sub_state

    @property
    def values(self) -> torch.Tensor:
        res = tuple()

        for sub_state_name in self._sub_states_order:
            sub_state: TimeseriesState = self[sub_state_name]
            res += (sub_state.values if any(sub_state.event_dim) else sub_state.values.unsqueeze(-1),)

        return torch.cat(res, dim=-1)

    def propagate_from(self, values: LazyTensor, time_increment=1.0):
        # NB: This is a hard assumption that the values are in the correct order...
        result = dict()

        last_ind = 0

        if callable(values):
            values = values()

        for sub_state_name in self._sub_states_order:
            sub_state: TimeseriesState = self[sub_state_name]

            dimension = sub_state.event_dim.numel()
            sub_values = values[..., last_ind : last_ind + dimension].squeeze(-1)

            result[sub_state_name] = sub_state.propagate_from(sub_values, time_increment=time_increment)
            last_ind += dimension

        return JointState(**result)


class StateSpaceModelState(JointState):
    """
    Implements a joint state for joint timeseries.
    """

    _ENFORCE_SAME_TIME_INDEX = False
