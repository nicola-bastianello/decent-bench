"""
Abstract class defining the Backend, and its methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
import inspect
from typing import Any, Callable

from ._array_creation import _BackendArrayCreation
from ._array_manipulation import _BackendArrayManipulation
from ._linalg import _BackendLinalg
from ._math import _BackendMath
from ._rng import _BackendRng



class _Backend(_BackendArrayCreation, _BackendArrayManipulation, _BackendLinalg, _BackendMath, _BackendRng, ABC):
    """
    Abstract class defining the methods each backend should implement.
    """



# create public functions as wrappers of the backend methods, with same signature/annotation/docs
def _create_wrapper(method_name: str) -> Callable:
    """Factory for one wrapper function."""
    abstract_method = getattr(_Backend, method_name)
    
    @wraps(abstract_method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        backend = get_backend()
        return getattr(backend, method_name)(*args, **kwargs)
    
    # preserve metadata
    wrapper.__signature__ = inspect.signature(abstract_method)
    wrapper.__annotations__ = abstract_method.__annotations__
    wrapper.__doc__ = abstract_method.__doc__
    
    return wrapper

# automatic generation of the API
def _expose_all_methods(abc_cls: type[_Backend]) -> None:
    """Auto-populate module globals with wrappers for all abstract methods."""
    for name, member in inspect.getmembers(abc_cls, predicate=inspect.isfunction):
        if getattr(member, '__isabstractmethod__', False):
            globals()[name] = _create_wrapper(name)

# at import time, the API is generated
_expose_all_methods(_Backend)

# expose the generated functions
__all__ = [
    name for name, member in inspect.getmembers(_Backend, predicate=inspect.isfunction)
    if getattr(member, '__isabstractmethod__', False)
]
