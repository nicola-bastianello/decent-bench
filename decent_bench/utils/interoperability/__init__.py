"""
Utilities for operating on arrays from different deep learning and linear algebra frameworks.

Mirrors NumPy's functionality for interoperability across frameworks.
"""

from __future__ import annotations

from . import _ext as ext
from ._decorators import autodecorate_cost_method
from ._functions import (
    all,  # noqa: A004
    argmax,
    argmin,
    astype,
    copy,
    diag,
    eye,
    eye_like,
    get_item,
    max,  # noqa: A004
    mean,
    min,  # noqa: A004
    norm,
    ones_like,
    rand_like,
    randn,
    randn_like,
    reshape,
    set_item,
    shape,
    squeeze,
    stack,
    sum,  # noqa: A004
    to_array,
    to_array_like,
    to_jax,
    to_numpy,
    to_tensorflow,
    to_torch,
    transpose,
    zeros,
    zeros_like,
)
from ._helpers import device_to_framework_device, framework_device_of_array, to_python_bool
from ._operators import (
    absolute,
    add,
    div,
    dot,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    logical_and,
    matmul,
    maximum,
    minimum,
    mul,
    negative,
    not_equal,
    power,
    sign,
    sqrt,
    sub,
)

__all__ = [  # noqa: RUF022
    # From _functions
    "all",
    "argmax",
    "argmin",
    "astype",
    "copy",
    "diag",
    "eye",
    "eye_like",
    "get_item",
    "max",
    "mean",
    "min",
    "norm",
    "ones_like",
    "rand_like",
    "randn",
    "randn_like",
    "reshape",
    "set_item",
    "shape",
    "squeeze",
    "stack",
    "sum",
    "to_array",
    "to_array_like",
    "to_numpy",
    "to_torch",
    "to_tensorflow",
    "to_jax",
    "transpose",
    "zeros",
    "zeros_like",
    # From _operators
    "absolute",
    "add",
    "div",
    "dot",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "matmul",
    "maximum",
    "minimum",
    "mul",
    "negative",
    "not_equal",
    "power",
    "sign",
    "sqrt",
    "sub",
    "to_python_bool",
    # From _helpers
    "device_to_framework_device",
    "framework_device_of_array",
    # From _decorators
    "autodecorate_cost_method",
    # Extensions
    "ext",
]
