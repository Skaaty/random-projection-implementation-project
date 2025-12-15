import numpy as np
from typing import Optional
import warnings


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
        normalize (bool): Whether to normalize input data before projection.
        components_ (np.ndarray): The learned random projection matrix after fitting.
        n_features_in_ (int): Number of features in input data.
        fitted_ (bool): Flag indicating whether the model has been fitted.
        n_iter_ (int): Number of iterations used for consistency with iterative algorithms (always 1 here).
        mean_ (np.ndarray | None): Mean of input features if normalize=True.
        std_ (np.ndarray | None): Standard deviation of input features if normalize=True.
    """

    def __init__(
        self,
        n_components: int,
        density: Optional[float] = None,
        normalize: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the RandomProjection object.

        Args:
            n_components (int): Target number of dimensions after projection.
            density (float | None, optional): Density for sparse projection in (0, 1]. Defaults to None.
            normalize (bool, optional): Whether to normalize input data (zero mean, unit variance). Defaults to False.
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
        self.normalize = normalize
        self.random_state = random_state
        self.components_ = None
        self.n_features_in_ = None
        self.fitted_ = False
        self.n_iter_ = None
        self.mean_ = None
        self.std_ = None

    @staticmethod
    def compute_min_components(n_samples: int, epsilon: float = 0.1) -> int:
        """
        Compute the minimum number of components required by the Johnson-Lindenstrauss lemma.

        For a dataset with n_samples, this gives the theoretical minimum dimension
        that preserves pairwise distances within (1±epsilon) with high probability.

        Args:
            n_samples (int): Number of samples in the dataset.
            epsilon (float, optional): Maximum distortion factor. Defaults to 0.1.

        Returns:
            int: Minimum number of components required.
        """
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("epsilon must be in (0, 1).")
        
        #Johnson-Lindenstrauss bound: k >= 4 * log(n) / (ε²/2 - ε³/3)
        k = 4 * np.log(n_samples) / (epsilon**2 / 2 - epsilon**3 / 3)
        return int(np.ceil(k))

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

        This implementation uses a more efficient vectorized approach while
        maintaining the same statistical properties.

        Args:
            n_features (int): Number of features in the input data.

        Returns:
            np.ndarray: Sparse random projection matrix of shape (n_features, n_components).

        Note:
            For very high dimensions (n_features * n_components > 1M), consider
            using sparse matrices for better memory efficiency.
        """
        if n_features * self.n_components > 1_000_000:
            warnings.warn(
                f"Projection matrix size {n_features}×{self.n_components} is large. "
                "Consider using sparse matrices for memory efficiency.",
                UserWarning
            )
        
        rng = np.random.default_rng(self.random_state)
        non_zero = int(np.ceil(self.density * n_features))
        scale = np.sqrt(1.0 / self.density) 
        
        components = np.zeros((n_features, self.n_components))
        
        all_indices = rng.choice(
            n_features, 
            size=(self.n_components, non_zero), 
            replace=False
        )
        
        for j in range(self.n_components):
            signs = rng.choice([-1, 1], size=non_zero)
            components[all_indices[j], j] = signs * scale
        
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

    def _normalize_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize input data to zero mean and unit variance.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            fit (bool): If True, compute normalization parameters from data.
                        If False, use previously computed parameters.

        Returns:
            np.ndarray: Normalized data.
        """
        if fit:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0, ddof=1)
            self.std_[self.std_ == 0] = 1.0
        
        return (X - self.mean_) / (self.std_ + 1e-8)

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
        
        if self.normalize:
            X = self._normalize_data(X, fit=True)
        
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
        
        if self.normalize and self.mean_ is not None and self.std_ is not None:
            X = (X - self.mean_) / (self.std_ + 1e-8)
        
        return X @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and project the data in one step.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Projected data of shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)

    def jl_quality(self, X: np.ndarray, n_pairs: int = 1000, epsilon: float = 0.1) -> float:
        """
        Evaluate the quality of the projection by checking the Johnson-Lindenstrauss property.

        Randomly samples pairs of points and checks if their distances are preserved
        within (1±epsilon) factor after projection.

        Args:
            X (np.ndarray): Original data of shape (n_samples, n_features).
            n_pairs (int, optional): Number of random pairs to test. Defaults to 1000.
            epsilon (float, optional): Maximum allowed distortion. Defaults to 0.1.

        Returns:
            float: Fraction of pairs that violate the distance preservation bound.
                   Lower is better (ideally close to 0).
        """
        if not self.fitted_:
            self.fit(X)
        
        X_proj = self.transform(X)
        n_samples = len(X)

        max_pairs = n_samples * (n_samples - 1) // 2
        n_pairs = min(n_pairs, max_pairs)

        rng = np.random.default_rng(self.random_state)
        all_pairs = np.array(np.triu_indices(n_samples, k=1)).T
        selected_indices = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        pairs = all_pairs[selected_indices]
        
        violations = 0
        for i, j in pairs:
            orig_dist = np.linalg.norm(X[i] - X[j])
            proj_dist = np.linalg.norm(X_proj[i] - X_proj[j])
            
            if orig_dist < 1e-10:
                ratio = 1.0
            else:
                ratio = proj_dist / orig_dist

            if ratio < 1 - epsilon or ratio > 1 + epsilon:
                violations += 1
        
        violation_rate = violations / n_pairs
        return violation_rate

    def __repr__(self):
        """Return string representation of the RandomProjection object."""
        type_proj = "sparse" if self.density else "dense"
        norm_str = ", normalized" if self.normalize else ""
        return (
            f"RandomProjection(n_components={self.n_components}, "
            f"type={type_proj}{norm_str}, "
            f"random_state={self.random_state})"
        )