from pathlib import Path

from decent_bench.utils.checkpoint_manager import CheckpointManager

if __name__ == "__main__":
    checkpoint_dir = Path(globals().get("CHECKPOINT_DIR", "benchmark_results/exp_1"))
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=500,
        keep_n_checkpoints=3,
    )
