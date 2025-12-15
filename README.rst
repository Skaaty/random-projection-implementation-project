.. These badges assume GitHub repository name and project name alignment.
.. Update <USER> if needed.

.. image:: https://api.cirrus-ci.com/github/konrad-wlodarczyk/random-projection-implementation-project.svg?branch=main
    :alt: Build Status
    :target: https://cirrus-ci.com/github/konrad-wlodarczyk/random-projection-implementation-project

.. image:: https://readthedocs.org/projects/random-projection-implementation-project/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://random-projection-implementation-project.readthedocs.io/en/latest/

.. image:: https://img.shields.io/coveralls/github/konrad-wlodarczyk/random-projection-implementation-project/main.svg
    :alt: Coveralls
    :target: https://coveralls.io/github/konrad-wlodarczyk/random-projection-implementation-project

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

====================================
Random Projection Implementation Project
====================================

A Python implementation of **random projection** — a dimensionality reduction technique that projects high-dimensional data into a lower-dimensional subspace using random mapping while approximately preserving pairwise distances. The implementation is intended for educational clarity, experimentation, and reproducible workflows. :contentReference[oaicite:0]{index=0}

Overview
========

Random projection provides a computationally efficient method for dimension reduction by multiplying the input data by a random matrix to produce a lower-dimensional representation. This approach is grounded in the Johnson–Lindenstrauss lemma, which guarantees approximate preservation of distances when projecting into a sufficiently large subspace. :contentReference[oaicite:1]{index=1}

Features
========

- Pure Python implementation
- Configurable projection dimensions
- Random projection matrix generation
- Support for fitting and transforming datasets
- Unit tests for core functionality

Project Structure
=================

::

    random-projection-implementation-project/
    ├── src/
    │   └── random_projection_implementation_project/
    │       ├── __init__.py
    │       └── random_projection.py
    │       └── random_projection_demo.py
    ├── tests/
    │   └── test_random_projection.py
    ├── docs/
    ├── README.rst
    ├── pyproject.toml
    ├── tox.ini
    └── requirements.txt

Installation
============

Clone the repository:

::

    git clone https://github.com/Skaaty/random-projection-implementation-project.git
    cd random-projection-implementation-project

Create a virtual environment and install dependencies:

::

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Usage
=====

Example usage:

::

    from random_projection_implementation_project.random_projection import RandomProjection

    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Create a projection to 2 dimensions
    rp = RandomProjection(n_components=2)
    rp.fit(X)

    X_transformed = rp.transform(X)

Testing
=======

Run the full test suite using:

::

    tox

or directly:

::

    pytest

Documentation
=============

Documentation is generated using **Sphinx** and **tox**.

Build the documentation locally:

::

    tox -e docs

The generated HTML documentation will be available in:

::

    docs/_build/html

Contributing
============

Contributions are welcome. Please ensure that:

- Code is well-documented
- Tests are added or updated where applicable
- All tox environments pass before submitting a pull request

License
=======

This project is licensed under the **MIT License**.  
See the ``LICENSE.txt`` file for details.

References
==========

- Bingham, E., & Mannila, H. Random projection in dimensionality reduction: Applications to image and text data (2001). :contentReference[oaicite:2]{index=2}

Note
====

This project has been set up using **PyScaffold 4.x**.  
For details and usage information see https://pyscaffold.org/.