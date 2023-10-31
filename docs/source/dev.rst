===============
Desenvolvimento
===============

Sobre
-----

Sobre o desenvolvimento da biblioteca


Dev Scripts
-----------

Clone:

.. code-block:: console

    git clone https://github.com/luizn22/auto-control-tools.git

Install requirements:

.. code-block:: console

    pip install -r requirements.txt
    pip install -r requirements_dev.txt

Install project as package:

.. code-block:: console

    pip install -e .

Run typing check:

.. code-block:: console

    mypy src

Run format check:

.. code-block:: console

    flake8 src

Run tests:

.. code-block:: console

    pytest

Build docs locally:

.. code-block:: console

    sphinx-build -b html .\docs\source\ .\rtd_build\
