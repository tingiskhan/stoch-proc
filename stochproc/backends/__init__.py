from .backend import Backend
from .torch import Torch
from .jax import Jax
from . import typing


class BackendDoesNotExist(Exception):
    pass


class BackendSelector(object):
    """
    Object for selecting backend to use.
    """

    _BACKEND_MAP = {
        "torch": Torch,
        "jax": Jax,
    }

    def __init__(self, default_backend: str):
        """
        Internal initializer for :class:`BackendSelector`.

        Args:
            default_backend (Backend): default backend to use.
        """

        self._current_backend = None
        self.change_backend(default_backend)
    
    @property
    def backend(self) -> Backend:
        return self._current_backend

    def change_backend(self, name: str):
        """
        Changes backend.

        Args:
            name (str): backend to change to.
        """

        if name not in self._BACKEND_MAP:
            raise BackendDoesNotExist(f"Currently does not support '{name}'!")

        self._current_backend = self._BACKEND_MAP[name]()


backend_selector = BackendSelector("torch")
backend = backend_selector.backend
