Utils
=====

This page highlights utility components that are commonly useful when running and debugging benchmarks.

Checkpoint Manager
------------------

Use :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` to:

- save benchmark progress
- resume interrupted runs
- persist computed metric outputs

.. code-block:: python

    from decent_bench.utils.checkpoint_manager import CheckpointManager

    checkpoint_manager = CheckpointManager(
        checkpoint_dir="benchmark_results/exp_1",
        checkpoint_step=500,
        keep_n_checkpoints=3,
    )

Network Utilities
-----------------

Use :mod:`~decent_bench.utils.network_utils` to visualize network topologies during debugging or reporting.

.. code-block:: python

    from decent_bench.utils import network_utils

    network_utils.plot_network(problem.network_structure, layout="circular", with_labels=True)

Algorithm Helpers and Interoperability
--------------------------------------

Two helper areas are especially useful for custom extensions:

- :mod:`~decent_bench.utils.algorithm_helpers` for common initialization patterns
- :mod:`~decent_bench.utils.interoperability` for framework-agnostic array/tensor operations

For the full catalog of utilities and signatures, see the API reference for:

- :doc:`decent_bench.utils <../api/decent_bench.utils>`
- :doc:`decent_bench.utils.interoperability <../api/decent_bench.utils.interoperability>`
