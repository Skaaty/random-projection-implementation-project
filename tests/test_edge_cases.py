"""
Edge-case tests for the RandomProjection class.

These tests verify numerical stability, behavior under identical inputs,
handling of extremely large or small values, and representation integrity.
"""

import numpy as np
from random_projection_implementation_project.random_projection import RandomProjection


def test_identical_points_dense_projection():
    """
    Test that RandomProjection handles identical samples correctly
    when using a dense projection matrix.

    Raises:
        AssertionError: If output shape is incorrect or contains non-finite values.
    """
    X = np.ones((20, 5))
    rp = RandomProjection(n_components=3, random_state=0)
    Xt = rp.fit_transform(X)

    assert Xt.shape == (20, 3)
    assert np.isfinite(Xt).all()


def test_large_values_projection():
    """
    Test numerical stability when projecting very large floating-point values.

    Raises:
        AssertionError: If projected values contain non-finite entries
                        or incorrect output shape.
    """
    X = np.array([
        [1e12, 1e12],
        [1e12 + 1, 1e12 + 1],
        [1e12 + 100, 1e12 + 100]
    ])
    rp = RandomProjection(n_components=2, random_state=0)
    Xt = rp.fit_transform(X)

    assert Xt.shape == (3, 2)
    assert np.isfinite(Xt).all()


def test_small_values_projection():
    """
    Test numerical stability when projecting extremely small floating-point values.

    Raises:
        AssertionError: If projected values contain non-finite entries
                        or incorrect output shape.
    """
    X = np.array([
        [1e-12, 1e-12],
        [2e-12, 2e-12],
        [3e-12, 3e-12]
    ])
    rp = RandomProjection(n_components=2)
    Xt = rp.fit_transform(X)

    assert Xt.shape == (3, 2)
    assert np.isfinite(Xt).all()


def test_repr_includes_configuration():
    """
    Test that the __repr__ method of RandomProjection returns a readable
    and informative string containing configuration parameters.

    Raises:
        AssertionError: If expected fields are missing from the representation.
    """
    rp = RandomProjection(n_components=4, density=None, random_state=10)
    rep = repr(rp)

    assert "RandomProjection" in rep
    assert "n_components=4" in rep
    assert "dense" in rep