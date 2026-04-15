from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedFrameworks, SupportedDevices



class _BackendArrayCreation(ABC):

    @abstractmethod
    def zeros(shape: tuple[int, ...]) -> Array:
        """
        Create a Array of zeros.

        Args:
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Array: Array of zeros.
        """

    @abstractmethod
    def zeros_like(array: Array) -> Array:
        """
        Create an array of zeros with the same shape and type as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Array of zeros in the same framework type as the input.
        """
    
    @abstractmethod
    def ones(shape: tuple[int, ...]) -> Array:
        """
        Create a Array of ones.

        Args:
            shape (tuple[int, ...]): Shape of the output array.

        Returns:
            Array: Array of ones.
        """

    @abstractmethod
    def ones_like(array: Array) -> Array:
        """
        Create an array of ones with the same shape and type as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Array of ones in the same framework type as the input.
        """

    @abstractmethod
    def eye(n: int) -> Array:
        """
        Create an identity matrix of size n x n in the specified framework.

        Args:
            n (int): Size of the identity matrix.

        Returns:
            Array: Identity matrix in the specified framework type.
        """
    
    @abstractmethod
    def eye_like(array: Array) -> Array:
        """
        Create an identity matrix with the same shape as the input.

        Args:
            array (Array): Input array.

        Returns:
            Array: Identity matrix in the same framework type as the input.
        """
