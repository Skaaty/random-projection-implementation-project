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

        #Config parameters
        self.n_components = n_components
        self.density = density  
        self.normalize = normalize
        self.random_state = random_state
        
        #Attributes initialized in fitting part
        self.components_ = None
        self.n_features_in_ = None
        self.fitted_ = False
        self.n_iter_ = None
        
        #Normalization statistics 
        self.mean_ = None
        self.std_ = None

    def _generate_dense_projection(self, n_features: int) -> np.ndarray:
        """Generate a dense Gaussian random projection matrix.

        Args:
            n_features (int): Number of features in the input data.

        Returns:
            np.ndarray: Random projection matrix of shape (n_features, n_components).
        """
        
        #Each entry ~ N(0, 1/sqrt(n_components))
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
        
        rng = np.random.default_rng(self.random_state)
        
        #Number pf non-zero entries per column
        non_zero = int(np.ceil(self.density * n_features))
        scale = np.sqrt(1.0 / self.density) 
        
        #Initialize zero matrix
        components = np.zeros((n_features, self.n_components))
        
        #randomly select indices for non-zero entries
        all_indices = rng.choice(
            n_features, 
            size=(self.n_components, non_zero), 
            replace=False
        )
        
        #assign random +-1 scaled values
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
            
            #Avoid division by zero for constant columns
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
        
        #Check that input is 2D
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        #Number of features
        self.n_features_in_ = X.shape[1]
        
        #Normalize if applicable
        if self.normalize:
            X = self._normalize_data(X, fit=True)
        
        #Generate random projection matrix based on the features
        self.components_ = self._initialize_projection(self.n_features_in_)
        
        #Mark as fitted
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
        
        #Apply normalization if used when fitting
        if self.normalize and self.mean_ is not None and self.std_ is not None:
            X = (X - self.mean_) / (self.std_ + 1e-8)
        
        #Matrix multiplication to reduce dimensionality
        return X @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model and project the data in one step.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Projected data of shape (n_samples, n_components).
        """
        #Combine the fit and transform process so that we are sure the data 
        #has been fitted before transforming
        return self.fit(X).transform(X)