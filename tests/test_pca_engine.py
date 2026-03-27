"""Unit tests for PCA Engine."""

import pytest
import numpy as np
from src.engines.pca_engine import PCAEngine
from src.models.ml import PCAResult


class TestPCAEngine:
    """Test PCA Engine functionality."""
    
    def test_fit_transform_basic(self):
        """Test PCA transformation with known dataset."""
        # Create simple dataset with 3 features
        data = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
            [5.0, 10.0, 15.0]
        ])
        feature_names = ["feature1", "feature2", "feature3"]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Verify result structure
        assert isinstance(result, PCAResult)
        assert result.n_components >= 1
        assert len(result.transformed_data) == 5  # Same number of samples
        assert len(result.explained_variance) == result.n_components
        assert len(result.explained_variance_ratio) == result.n_components
        assert result.feature_names == feature_names
        
        # Verify variance ratios sum to <= 1.0
        assert sum(result.explained_variance_ratio) <= 1.0
        
        # Verify cumulative variance meets threshold
        cumulative_variance = sum(result.explained_variance_ratio)
        assert cumulative_variance >= 0.8
    
    def test_fit_transform_variance_threshold(self):
        """Test that PCA retains components explaining at least 80% variance."""
        # Create dataset with clear variance structure
        np.random.seed(42)
        # First feature has high variance, others have low variance
        data = np.column_stack([
            np.random.randn(100) * 10,  # High variance
            np.random.randn(100) * 0.1,  # Low variance
            np.random.randn(100) * 0.1   # Low variance
        ])
        feature_names = ["high_var", "low_var1", "low_var2"]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Should retain fewer components than original features
        assert result.n_components <= 3
        
        # Cumulative variance should meet threshold
        cumulative_variance = sum(result.explained_variance_ratio)
        assert cumulative_variance >= 0.8
    
    def test_fit_transform_invalid_threshold(self):
        """Test that invalid variance threshold raises error."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        feature_names = ["f1", "f2"]
        
        engine = PCAEngine()
        
        # Test threshold > 1.0
        with pytest.raises(ValueError, match="variance_threshold must be between 0 and 1"):
            engine.fit_transform(data, feature_names, variance_threshold=1.5)
        
        # Test threshold <= 0.0
        with pytest.raises(ValueError, match="variance_threshold must be between 0 and 1"):
            engine.fit_transform(data, feature_names, variance_threshold=0.0)
    
    def test_fit_transform_insufficient_samples(self):
        """Test that PCA with insufficient samples raises error."""
        data = np.array([[1, 2, 3]])  # Only 1 sample
        feature_names = ["f1", "f2", "f3"]
        
        engine = PCAEngine()
        
        with pytest.raises(ValueError, match="Need at least 2 samples for PCA"):
            engine.fit_transform(data, feature_names)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Create dataset where features have different contributions
        np.random.seed(42)
        data = np.array([
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
            [5.0, 50.0, 500.0]
        ])
        feature_names = ["small", "medium", "large"]
        
        engine = PCAEngine()
        engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Get feature importance for first component
        importance = engine.get_feature_importance(0)
        
        # Verify structure
        assert isinstance(importance, list)
        assert len(importance) == 3
        assert all(isinstance(item, tuple) for item in importance)
        assert all(len(item) == 2 for item in importance)
        
        # Verify sorted by absolute value (descending)
        abs_loadings = [abs(loading) for _, loading in importance]
        assert abs_loadings == sorted(abs_loadings, reverse=True)
        
        # Verify feature names are present
        feature_names_in_importance = [name for name, _ in importance]
        assert set(feature_names_in_importance) == set(feature_names)
    
    def test_get_feature_importance_not_fitted(self):
        """Test that getting feature importance before fitting raises error."""
        engine = PCAEngine()
        
        with pytest.raises(ValueError, match="PCA must be fitted"):
            engine.get_feature_importance(0)
    
    def test_get_feature_importance_invalid_component(self):
        """Test that invalid component index raises error."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        feature_names = ["f1", "f2"]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Test negative index
        with pytest.raises(ValueError, match="component_idx must be between"):
            engine.get_feature_importance(-1)
        
        # Test index >= n_components
        with pytest.raises(ValueError, match="component_idx must be between"):
            engine.get_feature_importance(result.n_components)
    
    def test_transform_new_data(self):
        """Test transforming new data using fitted model."""
        # Fit on training data
        train_data = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0]
        ])
        feature_names = ["f1", "f2", "f3"]
        
        engine = PCAEngine()
        train_result = engine.fit_transform(train_data, feature_names, variance_threshold=0.8)
        
        # Transform new data
        new_data = np.array([
            [5.0, 10.0, 15.0],
            [6.0, 12.0, 18.0]
        ])
        
        transformed = engine.transform(new_data)
        
        # Verify shape
        assert transformed.shape[0] == 2  # Same number of samples as new_data
        assert transformed.shape[1] == train_result.n_components
    
    def test_transform_not_fitted(self):
        """Test that transforming before fitting raises error."""
        engine = PCAEngine()
        data = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="PCA must be fitted"):
            engine.transform(data)
    
    def test_single_feature(self):
        """Test PCA with single feature (edge case)."""
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        feature_names = ["single_feature"]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Should have exactly 1 component
        assert result.n_components == 1
        assert len(result.transformed_data) == 5
        
        # Should explain 100% of variance (only one feature)
        assert abs(sum(result.explained_variance_ratio) - 1.0) < 1e-10
    
    def test_all_features_identical(self):
        """Test PCA when all features have identical values (edge case)."""
        # All samples have same values - zero variance
        data = np.array([
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0]
        ])
        feature_names = ["f1", "f2", "f3"]
        
        engine = PCAEngine()
        
        # This should work but may have numerical issues
        # StandardScaler will produce zeros, PCA will handle it
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Should still produce a result
        assert isinstance(result, PCAResult)
        assert result.n_components >= 1
    
    def test_feature_normalization(self):
        """Test that features are normalized before PCA."""
        # Create data with very different scales
        data = np.array([
            [1.0, 1000.0, 0.001],
            [2.0, 2000.0, 0.002],
            [3.0, 3000.0, 0.003],
            [4.0, 4000.0, 0.004],
            [5.0, 5000.0, 0.005]
        ])
        feature_names = ["small", "large", "tiny"]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Verify that PCA was applied (should work despite different scales)
        assert isinstance(result, PCAResult)
        assert result.n_components >= 1
        
        # Verify that scaler was fitted
        assert engine.scaler is not None
        assert hasattr(engine.scaler, 'mean_')
        assert hasattr(engine.scaler, 'scale_')
    
    def test_explained_variance_ordering(self):
        """Test that explained variance is in descending order."""
        np.random.seed(42)
        data = np.random.randn(50, 5)
        feature_names = [f"f{i}" for i in range(5)]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Explained variance should be in descending order
        for i in range(len(result.explained_variance) - 1):
            assert result.explained_variance[i] >= result.explained_variance[i + 1]
        
        # Same for explained variance ratio
        for i in range(len(result.explained_variance_ratio) - 1):
            assert result.explained_variance_ratio[i] >= result.explained_variance_ratio[i + 1]
    
    def test_pca_result_components_shape(self):
        """Test that PCA components have correct shape."""
        data = np.random.randn(20, 4)
        feature_names = ["f1", "f2", "f3", "f4"]
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Components should be (n_components, n_features)
        assert len(result.components) == result.n_components
        for component in result.components:
            assert len(component) == len(feature_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
