"""
Tests for fit, transform, and fit_transform methods of RandomProjection.

These tests ensure correct attribute initialization, shape consistency,
and equivalence between fit/transform and fit_transform sequences.
"""

import numpy as np
from random_projection_implementation_project.random_projection import RandomProjection


def test_fit_sets_expected_attributes():
    """
    Test that calling fit initializes model attributes correctly.

    Raises:
        AssertionError: If attributes are missing or incorrect.
    """
    X = np.random.rand(30, 6)
    rp = RandomProjection(n_components=4)
    rp.fit(X)

    assert rp.fitted_ is True
    assert rp.n_features_in_ == 6
    assert rp.components_.shape == (6, 4)
    assert rp.n_iter_ == 1


def test_transform_output_shape():
    """
    Test that transform produces output with shape
    (n_samples, n_components).

    Raises:
        AssertionError: If the output array shape is incorrect.
    """
    X = np.random.rand(12, 5)
    rp = RandomProjection(n_components=3)
    rp.fit(X)

    Xt = rp.transform(X)
    assert Xt.shape == (12, 3)


def test_fit_transform_equivalence():
    """
    Test that fit_transform returns the same result as calling fit
    followed by transform on the same data.

    Raises:
        AssertionError: If results differ between both procedures.
    """
    X = np.random.rand(15, 4)

    rp1 = RandomProjection(n_components=2, random_state=0)
    out1 = rp1.fit_transform(X)

    rp2 = RandomProjection(n_components=2, random_state=0)
    rp2.fit(X)
    out2 = rp2.transform(X)

    assert np.allclose(out1, out2)