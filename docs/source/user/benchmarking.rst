Benchmarking
============

This page covers the standard benchmark workflow and the most important settings.

Executing a benchmark
---------------------

A typical run has three phases:

1. Execute algorithms with :func:`~decent_bench.benchmark.benchmark`.
2. Compute metrics with :func:`~decent_bench.benchmark.compute_metrics`.
3. Display or save outputs with :func:`~decent_bench.benchmark.display_metrics`.

.. code-block:: python

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD, ADMM

    if __name__ == "__main__":
        problem = benchmark.create_regression_problem(LinearRegressionCost)

        result = benchmark.benchmark(
            algorithms=[
                DGD(iterations=1000, step_size=0.01),
                ADMM(iterations=1000, rho=10.0, alpha=0.3),
            ],
            benchmark_problem=problem,
            n_trials=3,
        )

        metrics = benchmark.compute_metrics(result)
        benchmark.display_metrics(metrics)

Customizing benchmark settings
------------------------------

The most commonly tuned settings are:

- ``n_trials`` and ``max_processes`` for statistical robustness and runtime
- ``progress_step`` and ``show_speed`` for execution feedback
- ``table_metrics`` and ``plot_metrics`` for output selection
- ``confidence_level`` for confidence intervals

.. code-block:: python

    from logging import INFO

    from decent_bench import benchmark
    from decent_bench.costs import LinearRegressionCost
    from decent_bench.distributed_algorithms import DGD

    if __name__ == "__main__":
        result = benchmark.benchmark(
            algorithms=[DGD(iterations=2000, step_size=0.01)],
            benchmark_problem=benchmark.create_regression_problem(LinearRegressionCost),
            n_trials=5,
            max_processes=1,
            progress_step=200,
            show_speed=True,
            log_level=INFO,
        )

        metrics = benchmark.compute_metrics(
            result,
            confidence_level=0.9,
            log_level=INFO,
        )

        benchmark.display_metrics(metrics, save_path="results")

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
