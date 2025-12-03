"""
Tests for dense (Gaussian) random projection generation.

These tests verify shape correctness, reproducibility,
and expected behavior of dense random matrices.
"""

import numpy as np
from random_projection_implementation_project.random_projection import RandomProjection


def test_dense_projection_shape():
    """
    Test that the dense projection matrix has the expected shape.

    Raises:
        AssertionError: If the projection matrix shape is incorrect.
    """
    rp = RandomProjection(n_components=4, density=None)
    mat = rp._generate_dense_projection(n_features=6)
    assert mat.shape == (6, 4)


def test_dense_projection_reproducibility():
    """
    Test reproducibility of dense projection matrices when using
    the same random_state.

    Raises:
        AssertionError: If generated matrices differ for identical seeds.
    """
    rp1 = RandomProjection(n_components=3, random_state=42)
    rp2 = RandomProjection(n_components=3, random_state=42)
    m1 = rp1._generate_dense_projection(10)
    m2 = rp2._generate_dense_projection(10)
    assert np.allclose(m1, m2)


def test_dense_projection_scaling():
    """
    Test that dense Gaussian entries have the correct variance scaling
    of 1 / sqrt(n_components).

    Raises:
        AssertionError: If empirical standard deviation deviates excessively
                        from the expected scale factor.
    """
    rp = RandomProjection(n_components=5, random_state=0)
    mat = rp._generate_dense_projection(5000)

    expected_std = 1.0 / np.sqrt(5)
    empirical_std = np.std(mat)

    # Tolerance accounts for randomness
    assert np.isclose(empirical_std, expected_std, atol=1e-2)