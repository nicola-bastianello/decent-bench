from __future__ import annotations

from abc import ABC, abstractmethod

from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedArrayTypes, ArrayKey



class _BackendMath(ABC):

    @abstractmethod
    def sum(  # noqa: A001
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Sum elements of an array.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to sum.
                If None, sums over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Summed value.
        """

    @abstractmethod
    def mean(
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute mean of array elements.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute the mean.
                If None, computes mean of flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Mean value.
        """

    @abstractmethod
    def min(  # noqa: A001
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute minimum of array elements.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute minimum.
                If None, finds minimum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Minimum value.
        """

    @abstractmethod
    def max(  # noqa: A001
        array: Array,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """
        Compute maximum of array elements.

        Args:
            array (Array): Input array.
            dim (int | tuple[int, ...] | None): Dimension or dimensions along which to compute maximum.
                If None, finds maximum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Maximum value.
        """


    @abstractmethod
    def add(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
        """
        Element-wise addition of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise addition in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def iadd[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
        """
        Element-wise in-place addition of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise in-place addition in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def sub(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
        """
        Element-wise subtraction of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise subtraction in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def isub[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
        """
        Element-wise in-place subtraction of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise in-place subtraction in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def mul(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
        """
        Element-wise multiplication of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise multiplication in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def imul[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
        """
        Element-wise in-place multiplication of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise in-place multiplication in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def div(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
        """
        Element-wise division of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise division in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def idiv[T: Array](array1: T, array2: Array | SupportedArrayTypes) -> T:
        """
        Element-wise in-place division of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise in-place division in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """


    @abstractmethod
    def power(array: Array | SupportedArrayTypes, p: float) -> Array:
        """
        Raise array to p power.

        Args:
            array (Array | SupportedArrayTypes): The tensor.
            p (float): The power.

        Returns:
            Array: The result of the operation.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def ipow[T: Array](array: T, p: float) -> T:
        """
        Element-wise in-place power of an array.

        Args:
            array (Array | SupportedArrayTypes): Input array.
            p (float): The power.

        Returns:
            Array: Result of element-wise in-place power in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported

        """

    @abstractmethod
    def negative(array: Array | SupportedArrayTypes) -> Array:
        """
        Negate array.

        Args:
            array (Array | SupportedArrayTypes): The tensor.

        Returns:
            Array: The negated tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def absolute(array: Array | SupportedArrayTypes) -> Array:
        """
        Return the absolute value of a tensor.

        Args:
            array (Array | SupportedArrayTypes): The tensor.

        Returns:
            Array: The absolute value tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def sqrt(array: Array | SupportedArrayTypes) -> Array:
        """
        Return the square root of a tensor.

        Args:
            array (Array | SupportedArrayTypes): The tensor.

        Returns:
            Array: The square root tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def sign(array: Array | SupportedArrayTypes) -> Array:
        """
        Return the sign of a tensor.

        Args:
            array (Array | SupportedArrayTypes): The tensor.

        Returns:
            Array: The sign tensor.

        Raises:
            TypeError: If the type is not supported.

        """

    @abstractmethod
    def maximum(array1: Array | SupportedArrayTypes, array2: Array | SupportedArrayTypes) -> Array:
        """
        Element-wise maximum of two arrays.

        Args:
            array1 (Array | SupportedArrayTypes): First input array.
            array2 (Array | SupportedArrayTypes): Second input array.

        Returns:
            Array: Result of element-wise maximum in the same framework type as the inputs.

        Raises:
            TypeError: if the framework type of the input arrays is unsupported
                or if the input arrays are not of the same framework type.

        """

    @abstractmethod
    def argmax(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        """
        Compute index of maximum value.

        Args:
            array (Array): Input array.
            dim (int | None): Dimension along which to find maximum. If None, finds maximum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Indices of maximum values.
        """

    @abstractmethod
    def argmin(array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        """
        Compute index of minimum value.

        Args:
            array (Array): Input array.
            dim (int | None): Dimension along which to find minimum. If None, finds minimum over flattened array.
            keepdims (bool): If True, retains reduced dimensions with length 1.

        Returns:
            Array: Indices of minimum values.
        """


    @abstractmethod
    def set_item(
        array: Array | SupportedArrayTypes,
        key: ArrayKey,
        value: Array | SupportedArrayTypes,
    ) -> None:
        """
        Set the item at the specified index of the array to the given value.

        Args:
            array (Array | SupportedArrayTypes): The tensor.
            key (ArrayKey): The key or index to set.
            value (Array | SupportedArrayTypes): The value to set.

        Raises:
            TypeError: If the type is not supported.
            NotImplementedError: If the operation is not supported due to immutability.

        """

    @abstractmethod
    def get_item(array: Array, key: ArrayKey) -> Array:
        """
        Get the item at the specified index of the array.

        Args:
            array (Array): The tensor.
            key (ArrayKey): The key or index to get.

        Returns:
            Array: The item at the specified index.

        """
