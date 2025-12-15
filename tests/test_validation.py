"""
Validation tests for the RandomProjection class.

These tests ensure the constructor and public methods raise appropriate
exceptions when provided invalid input values or improperly shaped data.
"""

import pytest
import numpy as np
from random_projection_implementation_project.random_projection import RandomProjection


def test_invalid_n_components():
    """
    Test that RandomProjection raises a ValueError when n_components <= 0.

    Raises:
        ValueError: If n_components is not strictly positive.
    """
    with pytest.raises(ValueError):
        RandomProjection(n_components=0)


def test_invalid_density_zero():
    """
    Test that RandomProjection raises a ValueError when density <= 0.

    Raises:
        ValueError: If density is not within the valid interval (0, 1].
    """
    with pytest.raises(ValueError):
        RandomProjection(n_components=3, density=0)


def test_invalid_density_above_one():
    """
    Test that RandomProjection raises a ValueError when density > 1.

    Raises:
        ValueError: If density is not within the valid interval (0, 1].
    """
    with pytest.raises(ValueError):
        RandomProjection(n_components=3, density=1.5)


def test_fit_raises_on_invalid_shape():
    """
    Test that fit raises a ValueError when input data X is not a 2D array.

    Raises:
        ValueError: If X.ndim != 2.
    """
    rp = RandomProjection(n_components=3)
    X = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        rp.fit(X)


def test_transform_raises_before_fit():
    """
    Test that transform raises a ValueError if called before fit.

    Raises:
        ValueError: If transform is invoked on an unfitted model.
    """
    rp = RandomProjection(n_components=3)
    X = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        rp.transform(X)


def test_transform_raises_on_mismatched_dimensions():
    """
    Test that transform raises a ValueError when input feature dimension
    does not match the dimension used during fitting.

    Raises:
        ValueError: If X.shape[1] differs from n_features_in_.
    """
    X = np.random.rand(10, 4)
    rp = RandomProjection(n_components=2)
    rp.fit(X)

    X_new = np.random.rand(5, 5)
    with pytest.raises(ValueError):
        rp.transform(X_new)