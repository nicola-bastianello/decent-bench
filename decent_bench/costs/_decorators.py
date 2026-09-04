from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from decent_array import Array
from decent_array import interoperability as iop

T = TypeVar("T", bound=Callable[..., Any])


def autodecorate_cost_method[T: Callable[..., Any]](superclass_method: T) -> Callable[[Callable[..., Any]], T]:
    """Adapt native cost methods to the public cost method contract."""
    try:
        return_type_annotation = superclass_method.__annotations__["return"]
    except (AttributeError, KeyError):
        return_type_annotation = None

    def decorator(func: Callable[..., Any]) -> T:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if args:
                x_like = args[0]
            elif "x" in kwargs:
                x_like = kwargs["x"]
            else:
                raise ValueError("First argument must be 'x' for autodecorate_cost_method to work.")

            new_args = [arg.value if isinstance(arg, Array) else arg for arg in args]
            new_kwargs = {key: value.value if isinstance(value, Array) else value for key, value in kwargs.items()}
            result = func(self, *new_args, **new_kwargs)

            if return_type_annotation is Array and isinstance(x_like, Array):
                return iop.from_numpy_like(iop.to_numpy(result), x_like)
            return result

        return cast("T", wrapper)

    return decorator
