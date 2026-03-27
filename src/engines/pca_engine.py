"""PCA Engine for dimensionality reduction."""

from typing import List, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.ml import PCAResult


class PCAEngine:
    """Engine for performing Principal Component Analysis on customer features."""
    
    def __init__(self):
        """Initialize PCA Engine with scaler and PCA model."""
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
        self._is_fitted = False
    
    def fit_transform(
        self,
        customer_features: np.ndarray,
        feature_names: List[str],
        variance_threshold: float = 0.8
    ) -> PCAResult:
        """
        Apply PCA to customer features.
        
        Retains components explaining at least variance_threshold of total variance.
        
        Args:
            customer_features: Feature matrix (n_samples, n_features)
            feature_names: Names of features in the same order as columns
            variance_threshold: Minimum cumulative variance to explain (default 0.8)
        
        Returns:
            PCAResult with transformed data, explained variance, and loadings
        
        Raises:
            ValueError: If variance_threshold is not between 0 and 1
        """
        if not 0.0 < variance_threshold <= 1.0:
            raise ValueError("variance_threshold must be between 0 and 1")
        
        if customer_features.shape[0] < 2:
            raise ValueError("Need at least 2 samples for PCA")
        
        # Store feature names
        self.feature_names = feature_names
        
        # Normalize features using StandardScaler
        scaled_features = self.scaler.fit_transform(customer_features)
        
        # First fit PCA with all components to determine how many we need
        pca_full = PCA()
        pca_full.fit(scaled_features)
        
        # Handle edge case where variance is zero or NaN
        explained_variance_ratio = pca_full.explained_variance_ratio_
        explained_variance_ratio = np.nan_to_num(explained_variance_ratio, nan=0.0)
        
        # Determine number of components needed for variance threshold
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # If all variance is zero, use 1 component
        if cumulative_variance[-1] == 0.0:
            n_components = 1
        else:
            n_components = int(np.argmax(cumulative_variance >= variance_threshold) + 1)
        
        # Ensure at least 1 component
        n_components = max(1, n_components)
        
        # Fit PCA with determined number of components
        self.pca = PCA(n_components=n_components)
        transformed_data = self.pca.fit_transform(scaled_features)
        
        self._is_fitted = True
        
        # Handle NaN values in explained variance ratio
        explained_variance_ratio_clean = np.nan_to_num(
            self.pca.explained_variance_ratio_, 
            nan=0.0
        )
        
        # Create PCAResult
        return PCAResult(
            transformed_data=transformed_data.tolist(),
            explained_variance=self.pca.explained_variance_.tolist(),
            explained_variance_ratio=explained_variance_ratio_clean.tolist(),
            components=self.pca.components_.tolist(),
            feature_names=feature_names,
            n_components=n_components
        )
    
    def get_feature_importance(
        self,
        component_idx: int
    ) -> List[Tuple[str, float]]:
        """
        Get feature loadings for a specific principal component.
        
        Args:
            component_idx: Index of the principal component (0-based)
        
        Returns:
            List of (feature_name, loading) tuples sorted by absolute value (descending)
        
        Raises:
            ValueError: If PCA is not fitted or component_idx is invalid
        """
        if not self._is_fitted or self.pca is None:
            raise ValueError("PCA must be fitted before getting feature importance")
        
        if component_idx < 0 or component_idx >= self.pca.n_components_:
            raise ValueError(
                f"component_idx must be between 0 and {self.pca.n_components_ - 1}"
            )
        
        # Get loadings for the specified component
        loadings = self.pca.components_[component_idx]
        
        # Create list of (feature_name, loading) tuples
        feature_loadings = list(zip(self.feature_names, loadings))
        
        # Sort by absolute value of loading (descending)
        feature_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_loadings
    
    def transform(
        self,
        customer_features: np.ndarray
    ) -> np.ndarray:
        """
        Transform new customer data using fitted PCA model.
        
        Args:
            customer_features: Feature matrix (n_samples, n_features)
        
        Returns:
            Transformed feature matrix in PCA space
        
        Raises:
            ValueError: If PCA is not fitted
        """
        if not self._is_fitted or self.pca is None:
            raise ValueError("PCA must be fitted before transforming new data")
        
        # Normalize using fitted scaler
        scaled_features = self.scaler.transform(customer_features)
        
        # Transform using fitted PCA
        return self.pca.transform(scaled_features)
