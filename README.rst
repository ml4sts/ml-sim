Fair-Sim:
=========
Synthetic Data generators for fairness evaluation

Getting Started
================



To use the package, download (or clone) and:

.. code-block:: bash

  cd fair-sim/
  pip install .

To use the package, after installed::

  import fair-sim




Development
============

To work in a separate dev

.. code-block:: bash

TOOD: make this true

To compile docs, sphinx and some extensions are required, all are included in
`requirements.txt` and can be installed with

.. code-block:: bash

  pip install -r requirements.txt

then

.. code-block:: bash

  cd docs/
  make html


To reinstall  package after changes

.. code-block:: bash

  pip install --upgrade .

When updating the package and working in a notebook, the notebook's kernel will
need to be restarted to get the updates, if they're done outside of the notebook.

(only needed in development or after upgrade)
