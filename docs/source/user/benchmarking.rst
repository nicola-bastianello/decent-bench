Benchmarking
============

TODO: This page covers the standard benchmark workflow and the most important settings.
:doc:`customizing <customizing>` will show how to customize each component of the benchmark

Running a benchmark
-------------------

A typical run has three phases:

1. Execute algorithms with :func:`~decent_bench.benchmark.benchmark`.
2. Compute metrics with :func:`~decent_bench.benchmark.compute_metrics`.
3. Display or save outputs with :func:`~decent_bench.benchmark.display_metrics`.

.. literalinclude:: ../../../test/user-guide/benchmarking_minimal.py
   :language: python

Customizing benchmark settings
------------------------------

The most commonly tuned settings are:

- ``n_trials`` and ``max_processes`` for statistical robustness and runtime
- ``progress_step`` and ``show_speed`` for execution feedback
- ``table_metrics`` and ``plot_metrics`` for output selection
- ``confidence_level`` for confidence intervals

.. literalinclude:: ../../../test/user-guide/benchmarking_custom_settings.py
   :language: python

Reproducibility (setting a seed)
--------------------------------

For reproducible experiments, set seeds consistently for all random sources you use.

- Python random module
- NumPy
- framework-specific RNGs (for example PyTorch)
- graph generation utilities that accept ``seed``

.. code-block:: python

    import random

    import numpy as np
    import networkx as nx

    random.seed(0)
    np.random.seed(0)

    graph = nx.random_regular_graph(d=3, n=100, seed=0)

If your benchmark uses framework-level randomness, also set that framework's seed at startup.
Use a fixed seed per experiment when comparing algorithms, and change the seed between experiment batches when you
want robustness checks.
