"""
Tests for sparse random projection generation.

These tests verify correct sparsity, reproducibility, and shape
for projections generated using the Achlioptas scheme.
"""

import numpy as np
from random_projection_implementation_project.random_projection import RandomProjection


def test_sparse_projection_shape():
    """
    Test that the sparse projection matrix has the expected shape.

    Raises:
        AssertionError: If resulting projection matrix shape is incorrect.
    """
    rp = RandomProjection(n_components=4, density=0.2)
    mat = rp._generate_sparse_projection(n_features=10)
    assert mat.shape == (10, 4)


def test_sparse_projection_nonzero_count():
    """
    Test that the number of non-zero coefficients per column matches
    ceil(density * n_features).

    Raises:
        AssertionError: If non-zero counts differ from expected values.
    """
    density = 0.25
    n_features = 40
    rp = RandomProjection(n_components=3, density=density, random_state=0)
    mat = rp._generate_sparse_projection(n_features)

    expected_nonzero = int(np.ceil(density * n_features))
    nonzero_per_col = (mat != 0).sum(axis=0)

    assert np.all(nonzero_per_col == expected_nonzero)


def test_sparse_projection_reproducibility():
    """
    Test reproducibility of sparse projection matrices when using
    identical random_state values.

    Raises:
        AssertionError: If results differ for identical seeds.
    """
    rp1 = RandomProjection(n_components=3, density=0.2, random_state=123)
    rp2 = RandomProjection(n_components=3, density=0.2, random_state=123)

    m1 = rp1._generate_sparse_projection(20)
    m2 = rp2._generate_sparse_projection(20)

    assert np.allclose(m1, m2)