Getting Started
===============
Want to contribute to decent-bench? That's great! This page covers development setup and available tooling.


Prerequisites
-------------
* `Python 3.13+ <https://www.python.org/downloads/>`_
* `tox <https://tox.wiki/en/stable/installation.html>`_

Installation for Development
-----------------------------
.. code-block::

   git clone https://github.com/team-decent/decent-bench.git
   cd decent-bench
   tox -e dev                           # create dev env (admin privileges may be needed)
   source .tox/dev/bin/activate         # activate dev env on Mac/Linux
   .\.tox\dev\Scripts\activate          # activate dev env on Windows

Optionally install development dependencies with proper gpu support, e.g. for PyTorch and TensorFlow:

.. code-block::

   tox -e dev-gpu

It is not recommended to use the development environments for regular usage of decent-bench, as they
contain additional packages that are not needed for that purpose. This may cause performance degradation
due to multiple packages competing for resources (e.g. GPU resources).

Tooling
-------
To make sure all GitHub status checks pass, simply run :code:`tox`. You can also run individual checks:

.. code-block::

    tox -e mypy       # find typing issues
    tox -e pytest     # run tests
    tox -e ruff       # find formatting and style issues
    tox -e sphinx     # rebuild documentation

Note: Running :code:`tox` commands can take several minutes and may require admin privileges. 
If you have mypy addon installed in your IDE, you can use it to get instant feedback on typing issues while coding.
If mypy fails with ``KeyError: 'setter_type'``, delete the ``.mypy_cache`` folder in the project root.

Tools can also be used directly (instead of via tox) after activating the dev environment. Useful examples include:

.. code-block::

    ruff check decent_bench --fix                           # find and fix style issues
    ruff format decent_bench                                # format code
    mypy decent_bench --strict                              # find typing issues
    pytest test                                             # run tests
    sphinx-build -W -E -b html docs/source docs/build/html  # rebuild html doc files

To verify that doc changes look good, use an html previewer such as
`Live Preview <https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server>`_.
If you are running :code:`pytest test` while using ``WSL`` on Windows and it starts to randomly fail (or if its really slow), restart your ``WSL`` instance.
