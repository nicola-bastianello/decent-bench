Introduction
============

Decentralized optimization trains models across multiple agents without sending raw training data to a central node.
This helps with privacy constraints, communication limits, and large-scale distributed systems.

In decent-bench, you can benchmark decentralized algorithms under realistic communication constraints such as:

- asynchronous participation
- compression
- noise
- packet drops
- sparse network topologies

The goal is to compare algorithm behavior in settings that are close to real deployments, not only idealized conditions.

Use this User Guide to:

- install decent-bench and verify requirements
- run and configure benchmarks
- customize networks, algorithms, costs, and metrics
- use utility helpers for checkpoints and reproducibility
