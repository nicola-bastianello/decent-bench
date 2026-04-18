import runpy
from pathlib import Path

from decent_bench.networks import P2PNetwork
from decent_bench.utils.checkpoint_manager import CheckpointManager


USER_GUIDE_DIR = Path(__file__).parent / "user-guide"


def _run_script(path: Path, globals_override: dict[str, object] | None = None) -> dict[str, object]:
    return runpy.run_path(str(path), run_name="__main__", init_globals=globals_override)


def test_minimal_benchmark_example_runs() -> None:
    module_globals = _run_script(
        USER_GUIDE_DIR / "benchmarking_minimal.py",
        globals_override={"ITERATIONS": 2, "N_AGENTS": 4, "N_TRIALS": 1},
    )

    benchmark_result = module_globals["benchmark_result"]
    metric_result = module_globals["metric_result"]

    assert metric_result.available_algorithms == ["DGD"]
    assert len(benchmark_result.states) == 1
    for trial_results in benchmark_result.states.values():
        assert len(trial_results) == 1
        for network in trial_results:
            for agent in network.agents():
                assert len(agent._x_history) == 3  # noqa: SLF001


def test_custom_problem_example_builds() -> None:
    module_globals = _run_script(USER_GUIDE_DIR / "customizing_network.py", globals_override={"N_AGENTS": 6})

    problem = module_globals["problem"]

    assert isinstance(problem.network, P2PNetwork)
    assert len(problem.network.agents()) == 6


def test_custom_settings_example_runs() -> None:
    module_globals = _run_script(
        USER_GUIDE_DIR / "benchmarking_custom_settings.py",
        globals_override={"ITERATIONS": 2, "N_AGENTS": 4, "N_TRIALS": 1},
    )

    metrics = module_globals["metric_result"]

    assert metrics.available_algorithms == ["DGD"]


def test_checkpoint_manager_example() -> None:
    checkpoint_dir = Path("tmp_test_checkpoint")
    if checkpoint_dir.exists():
        for child in checkpoint_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_dir():
                        nested.rmdir()
                child.rmdir()
        checkpoint_dir.rmdir()

    module_globals = _run_script(
        USER_GUIDE_DIR / "utils_checkpoint.py",
        globals_override={"CHECKPOINT_DIR": checkpoint_dir},
    )

    checkpoint_manager = module_globals["checkpoint_manager"]

    assert isinstance(checkpoint_manager, CheckpointManager)
    assert checkpoint_manager.checkpoint_dir.exists()
    assert checkpoint_manager.is_empty()

    checkpoint_manager.checkpoint_dir.rmdir()
