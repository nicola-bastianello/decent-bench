Utils
=====

This page highlights utility components that are commonly useful when running and debugging benchmarks.

Checkpoint Manager
------------------

Use :class:`~decent_bench.utils.checkpoint_manager.CheckpointManager` to:

- save benchmark progress
- resume interrupted runs
- persist computed metric outputs

.. literalinclude:: ../../../test/user-guide/utils_checkpoint.py
   :language: python

Network Utilities
-----------------

Use :mod:`~decent_bench.utils.network_utils` to visualize network topologies during debugging or reporting.

.. literalinclude:: ../../../test/user-guide/utils_network_plot.py
    :language: python

Algorithm Helpers and Interoperability
--------------------------------------

Two helper areas are especially useful for custom extensions:

- :mod:`~decent_bench.utils.algorithm_helpers` for common initialization patterns
- :mod:`~decent_bench.utils.interoperability` for framework-agnostic array/tensor operations

For the full catalog of utilities and signatures, see the API reference for:

- :doc:`decent_bench.utils <../api/decent_bench.utils>`
- :doc:`decent_bench.utils.interoperability <../api/decent_bench.utils.interoperability>`
