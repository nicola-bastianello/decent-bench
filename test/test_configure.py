import json
from pathlib import Path

import pytest
from decent_array.types import Devices, Frameworks

from decent_bench.benchmark import configure
from decent_bench.benchmark._configure import get_checkpoint_manager, get_config


def test_configure_sets_process_state_and_checkpoint_options(tmp_path: Path) -> None:  # noqa: D103
    storage_dir = tmp_path / "experiment"

    configure(
        Frameworks.NUMPY,
        Devices.CPU,
        seed=42,
        log_level=30,
        storage_dir=storage_dir,
        n_checkpoints=5,
        compression_level=2,
    )

    config = get_config()
    manager = get_checkpoint_manager()
    assert config.framework is Frameworks.NUMPY
    assert config.device is Devices.CPU
    assert config.seed == 42
    assert config.log_level == 30
    assert config.storage_dir == storage_dir
    assert config.n_checkpoints == 5
    assert config.compression_level == 2
    assert manager is not None
    assert manager.checkpoint_dir == storage_dir
    assert manager.n_checkpoints == 5
    assert manager.compression_level == 2


def test_configure_requires_backend_for_new_experiment() -> None:  # noqa: D103
    with pytest.raises(ValueError, match="framework and device must be provided"):
        configure()


def test_configure_without_storage_disables_checkpointing() -> None:  # noqa: D103
    configure(Frameworks.NUMPY, Devices.CPU)

    assert get_checkpoint_manager() is None


def test_configure_rejects_second_call() -> None:  # noqa: D103
    configure(Frameworks.NUMPY, Devices.CPU)

    with pytest.raises(RuntimeError, match="only be called once"):
        configure(Frameworks.NUMPY, Devices.CPU)


def test_configure_loads_existing_experiment_settings(tmp_path: Path) -> None:  # noqa: D103
    storage_dir = tmp_path / "experiment"
    storage_dir.mkdir()
    (storage_dir / "metadata.json").write_text(
        json.dumps(
            {
                "backend": {"framework": Frameworks.NUMPY.value, "device": Devices.CPU.value},
                "checkpointing": {"n_checkpoints": 7, "compression_level": 3},
                "rng_seed": 123,
            }
        ),
        encoding="utf-8",
    )

    configure(storage_dir=storage_dir, seed=999)

    config = get_config()
    manager = get_checkpoint_manager()
    assert config.framework is Frameworks.NUMPY
    assert config.device is Devices.CPU
    assert config.seed == 123
    assert config.n_checkpoints == 7
    assert config.compression_level == 3
    assert manager is not None
    assert manager.n_checkpoints == 7
    assert manager.compression_level == 3


def test_configure_rejects_backend_for_existing_experiment(tmp_path: Path) -> None:  # noqa: D103
    storage_dir = tmp_path / "experiment"
    storage_dir.mkdir()
    (storage_dir / "metadata.json").write_text(
        json.dumps({"backend": {"framework": Frameworks.NUMPY.value, "device": Devices.CPU.value}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not be provided"):
        configure(Frameworks.NUMPY, Devices.CPU, storage_dir=storage_dir)
