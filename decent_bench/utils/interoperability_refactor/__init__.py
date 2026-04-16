"""
Utilities for operating on arrays from different deep learning and linear algebra frameworks.

Mirrors NumPy's functionality for interoperability across frameworks.
"""

from __future__ import annotations

from ._decorators import autodecorate_cost_method

from ._abstract import (
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
    reshape,
    set_item,
    shape,
    squeeze,
    stack,
    sum,  # noqa: A004
    # to_array,
    # to_array_like,
    # to_jax,
    # to_numpy,
    # to_tensorflow,
    # to_torch,
    transpose,
    zeros,
    zeros_like,
    absolute,
    add,
    div,
    dot,
    matmul,
    maximum,
    mul,
    negative,
    power,
    sign,
    sqrt,
    sub,
)

from ._helpers import device_to_framework_device, framework_device_of_array

from ._rng import (
    get_rng_state,
    get_seed,
    rng_jax,
    rng_numpy,
    rng_tensorflow,
    rng_torch,
    set_rng_state,
    set_seed,
)

__all__ = [  # noqa: RUF022
    # From _abstract
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
    "reshape",
    "set_item",
    "shape",
    "squeeze",
    "stack",
    "sum",
    # "to_array",
    # "to_array_like",
    # "to_numpy",
    # "to_torch",
    # "to_tensorflow",
    # "to_jax",
    "transpose",
    "zeros",
    "zeros_like",
    "absolute",
    "add",
    "div",
    "dot",
    "matmul",
    "maximum",
    "mul",
    "negative",
    "power",
    "sign",
    "sqrt",
    "sub",
    # From _helpers
    "device_to_framework_device",
    "framework_device_of_array",
    # From _decorators
    "autodecorate_cost_method",
    # RNG manager
    "choice",
    "rng_numpy",
    "get_rng_state",
    "get_seed",
    "rng_tensorflow",
    "rng_torch",
    "set_rng_state",
    "set_seed",
    "uniform_like",
    "uniform",
    "normal",
    "normal_like",
    "rng_jax",
]
