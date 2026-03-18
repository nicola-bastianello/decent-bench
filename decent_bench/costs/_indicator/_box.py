import decent_bench.utils.interoperability as iop
from decent_bench.utils.array import Array
from ._indicator_cost import IndicatorCost
from decent_bench.utils.types import SupportedArrayTypes


class BoxSetIndicator(IndicatorCost):
    r"""Indicator cost for a box set :math:`\mathbb{X} = [\mathbf{\ell}, \mathbf{u}]`."""

    def __init__(
        self,
        lower_bound: Array | SupportedArrayTypes,
        upper_bound: Array | SupportedArrayTypes,
    ):
        if iop.shape(lower_bound) != iop.shape(upper_bound):  # type: ignore[arg-type]
            raise ValueError("`lower_bound` and `upper_bound` must have the same shape.")
        framework_lower, device_lower = iop.framework_device_of_array(lower_bound)  # type: ignore[arg-type]
        framework_upper, device_upper = iop.framework_device_of_array(upper_bound)  # type: ignore[arg-type]
        if framework_lower != framework_upper or device_lower != device_upper:
            raise ValueError("`lower_bound` and `upper_bound` must have the framework and device")

        super().__init__(iop.shape(lower_bound), framework=framework_lower, device=device_lower)  # type: ignore[arg-type]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def projection(self, x: Array) -> Array:
        """Projection of `x` onto the convex set that defines the indicator cost."""
        return iop.maximum(self.lower_bound, iop.minimum(self.upper_bound, x))

    def belongs_to_set(self, x: Array) -> bool:
        """Method checking whether `x` belongs to the convex set that defines the indicator cost."""
        return bool(iop.all(x >= self.lower_bound) and iop.all(x <= self.upper_bound))
