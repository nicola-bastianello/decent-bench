User Guide
==========
This user guide provides a concise, up-to-date path for installing, benchmarking, and customizing decent-bench.

Install
-------

Requirements:

- `Python 3.13+ <https://www.python.org/downloads/>`_
- `pip <https://pip.pypa.io/en/stable/installation/>`_ (normally included with Python)
- Optional: a virtual environment tool such as ``venv``

For GPU-backed workflows, install framework builds (for example PyTorch or TensorFlow) compatible with your
CUDA/ROCm setup.

.. code-block:: bash

   pip install decent-bench

.. toctree::
   :maxdepth: 1

   introduction
   benchmarking
   customizing
   utils
