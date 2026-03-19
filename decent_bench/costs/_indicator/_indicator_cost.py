from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks


class IndicatorCost(Cost, ABC):
    r"""
    Indicator cost for a convex set :math:`\mathbb{X}`.

    .. math:: f(\mathbf{x}) = \iota_{\mathbb{X}}(\mathbf{x}) = \begin{cases}
                0 & \text{if} \ \mathbf{x} \in \mathbb{X} \\
                +\infty & \text{otherwise}
            \end{cases}

    Implementing the `belongs_to_set` and `projection` methods fully defines the indicator cost function. Note that
    the cost is not differentiable (gradient and Hessian do not exist), and that the proximal coincides with the
    projection onto the set.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
    ):
        if len(shape) == 0:
            raise ValueError("Cost shape must be non-empty.")
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"Cost shape must be positive, got {shape}.")
        self._shape = shape
        self._framework = framework
        self._device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def framework(self) -> SupportedFrameworks:
        return self._framework

    @property
    def device(self) -> SupportedDevices:
        return self._device

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return np.nan

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        return 0

    @iop.autodecorate_cost_method(Cost.function)
    def function(self, x: Array) -> float:
        """Evaluate function at x."""
        return 0 if self.belongs_to_set(x) else np.inf

    @iop.autodecorate_cost_method(Cost.gradient)
    def gradient(self, x: Array) -> Array:
        raise NotImplementedError("Indicator functions are not differentiable.")

    @iop.autodecorate_cost_method(Cost.hessian)
    def hessian(self, x: Array) -> Array:
        raise NotImplementedError("Indicator functions are not differentiable.")

    @iop.autodecorate_cost_method(Cost.proximal)
    def proximal(self, x: Array, _: float = 1) -> Array:
        r"""Proximal at x, which coincides with a projection onto the convex set that defines the indicator cost."""
        return self.projection(x)

    @abstractmethod
    def projection(self, x: Array) -> Array:
        """Projection of `x` onto the convex set that defines the indicator cost."""

    @abstractmethod
    def belongs_to_set(self, x: Array) -> bool:
        """Check whether `x` belongs to the convex set that defines the indicator cost."""
