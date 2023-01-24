import numpy as np
from typing import TypeVar, Generic, Tuple


TArray = TypeVar("TArray")
TDistribution = TypeVar("TDistribution")


class Backend(Generic[TArray, TDistribution]):
    """
    Defines a base class for backends.
    """

    def array(self, x, dtype: str = None):
        """
        Coerce item to array.
        
        Args:
            x (): item to cast as an array.
            dtype (str): dtype to use. Defaults to `None`.
        """

        raise NotImplementedError()

    def coerce_arrays(self, *x) -> Tuple[TArray, ...]:
        """
        Coerces objects to be :attr:`TArray` and verifies that all are on the same device.

        Returns:
            Tuple[TArray, ...]: Input parameters coerced to tensors.
        """

        raise NotImplementedError()
    
    def affine_transform(self, base: TDistribution, loc: TArray, scale: TArray, n_dim: int) -> TDistribution:
        """
        Utility method for constructing an affine transformed distribution.

        Args:
            base (TDistribution): base distribution.
            loc (TArray): location.
            scale (TArray): scale.

        Returns:
            TDistribution: transformed distribution.
        """

        raise NotImplementedError()

    def broadcast_arrays(self, *x: TArray) -> Tuple[TArray, ...]:
        """
        Broadcasts arrays to same shape.

        Returns:
            Tuple[TArray, ...]: broadcasted arrays.
        """

        raise NotImplementedError()
