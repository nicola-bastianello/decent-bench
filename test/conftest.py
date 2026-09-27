import os
import importlib

import pytest
from decent_array import interoperability as iop
from decent_array.types import Devices, Frameworks


@pytest.fixture(scope="session", autouse=True)
def activate_backend() -> None:
    """Activate the backend selected for the pytest session."""
    backend = Frameworks(os.environ.get("DECENT_BENCH_BACKEND", Frameworks.NUMPY.value))
    device = Devices(os.environ.get("DECENT_BENCH_DEVICE", Devices.CPU.value))
    iop.set_backend(backend, device)


@pytest.fixture(autouse=True)
def reset_benchmark_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a fresh configure-once state."""
    configure_module = importlib.import_module("decent_bench.benchmark._configure")
    monkeypatch.setattr(configure_module, "_STATE", configure_module._ConfigurationState())
