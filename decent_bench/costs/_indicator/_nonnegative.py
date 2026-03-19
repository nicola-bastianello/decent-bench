import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.utils.array import Array
from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._indicator_cost import IndicatorCost


class NonnegativeIndicator(IndicatorCost):
    r"""Indicator cost for the nonnegative orthant."""

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        framework: SupportedFrameworks = SupportedFrameworks.NUMPY,
        device: SupportedDevices = SupportedDevices.CPU,
    ):
        super().__init__(shape, framework=framework, device=device)

    def projection(self, x: Array) -> Array:
        """Projection of `x` onto the convex set that defines the indicator cost."""
        return iop.maximum(x, 0)

    def belongs_to_set(self, x: Array) -> bool:
        """Check whether `x` belongs to the convex set that defines the indicator cost."""
        return iop.to_python_bool(iop.all(x >= 0))

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, NonnegativeIndicator):
            return self

        return SumCost([self, other])
