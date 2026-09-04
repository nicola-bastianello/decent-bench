import os

import pytest
from decent_array import interoperability as iop
from decent_array.types import Devices, Frameworks


@pytest.fixture(scope="session", autouse=True)
def activate_backend() -> None:
    """Activate the backend selected for the pytest session."""
    backend = Frameworks(os.environ.get("DECENT_BENCH_BACKEND", Frameworks.NUMPY.value))
    device = Devices(os.environ.get("DECENT_BENCH_DEVICE", Devices.CPU.value))
    iop.set_backend(backend, device)
