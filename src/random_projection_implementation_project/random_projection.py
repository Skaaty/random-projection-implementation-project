import numpy as np
from typing import Optional

class RandomProjection:
    """Custom implementation of the Random Projection algorithm for dimensionality reduction.

    This implementation supports both dense (Gaussian) and sparse random projections.
    It can be used to reduce the dimensionality of high-dimensional datasets
    while approximately preserving pairwise distances between points, based on
    the Johnson-Lindenstrauss lemma.

    Attributes:
        n_components (int): Target dimensionality after projection.
        density (float | None): Density of the sparse projection. If None, a dense Gaussian projection is used.
        random_state (Optional[int]): Seed for reproducibility.
        components_ (np.ndarray): The learned random projection matrix after fitting.
        n_features_in_ (int): Number of features in input data.
        fitted_ (bool): Flag indicating whether the model has been fitted.
        n_iter_ (int): Number of iterations used for consistency with iterative algorithms (always 1 here).
    """

    def __init__(
        self,
        n_components: int,
        density: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the RandomProjection object.

        Args:
            n_components (int): Target number of dimensions after projection.
            density (float | None, optional): Density for sparse projection in (0, 1]. Defaults to None.
            random_state (Optional[int], optional): Random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: If n_components <= 0 or density is not in (0, 1].
        """
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        if density is not None and (density <= 0 or density > 1):
            raise ValueError("density must be in (0, 1].")

        self.n_components = n_components
        self.density = density
        self.random_state = random_state
        self.components_ = None
        self.n_features_in_ = None
        self.fitted_ = False
        self.n_iter_ = None

    def _generate_dense_projection(self, n_features: int) -> np.ndarray:
        """Generate a dense Gaussian random projection matrix.

        Args:
            n_features (int): Number of features in the input data.

        Returns:
            np.ndarray: Random projection matrix of shape (n_features, n_components).
        """
        rng = np.random.default_rng(self.random_state)
        return rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.n_components),
            size=(n_features, self.n_components)
        )

    def _generate_sparse_projection(self, n_features: int) -> np.ndarray:
        """Generate a sparse random projection matrix using Achlioptas scheme.

        Args:
            n_features (int): Number of features in the input data.

        Returns:
            np.ndarray: Sparse random projection matrix of shape (n_features, n_components).
        """
        rng = np.random.default_rng(self.random_state)
        components = np.zeros((n_features, self.n_components))
        non_zero = int(np.ceil(self.density * n_features))
        scale = np.sqrt(1.0 / self.density)

        for j in range(self.n_components):
            indices = rng.choice(n_features, size=non_zero, replace=False)
            signs = rng.choice([-1, 1], size=non_zero)
            components[indices, j] = signs * scale
        return components

    def _initialize_projection(self, n_features: int) -> np.ndarray:
        """Initialize the random projection matrix, dense or sparse.

        Args:
            n_features (int): Number of features in the input data.

        Returns:
            np.ndarray: Initialized random projection matrix.
        """
        if self.density is None:
            return self._generate_dense_projection(n_features)
        else:
            return self._generate_sparse_projection(n_features)

    def fit(self, X: np.ndarray):
        """Fit the Random Projection model to input data.

        This method generates the random projection matrix based on the input
        feature dimension.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Raises:
            ValueError: If X is not a 2D array.

        Returns:
            self: Fitted RandomProjection object.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        self.n_features_in_ = X.shape[1]
        self.components_ = self._initialize_projection(self.n_features_in_)
        self.fitted_ = True
        self.n_iter_ = 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project input data into lower-dimensional space.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Raises:
            ValueError: If model is not fitted or input dimension does not match.

        Returns:
            np.ndarray: Transformed data of shape (n_samples, n_components).
        """
        if not self.fitted_:
            raise ValueError("Model has not been fitted. Call fit(X) first.")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input features {X.shape[1]} do not match fitted dimension {self.n_features_in_}."
            )
        return X @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and project the data in one step.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Projected data of shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)

    def __repr__(self):
        """Return string representation of the RandomProjection object."""
        type_proj = "sparse" if self.density else "dense"
        return (
            f"RandomProjection(n_components={self.n_components}, "
            f"type={type_proj}, "
            f"random_state={self.random_state})"
        )
