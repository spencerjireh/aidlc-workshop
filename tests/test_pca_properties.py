"""Property-based tests for PCA Engine."""

import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from src.engines.pca_engine import PCAEngine


# Custom strategy for generating valid customer feature data
@st.composite
def customer_feature_data(draw, min_samples=10, max_samples=100, min_features=2, max_features=10):
    """Generate valid customer feature data for PCA testing."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_features = draw(st.integers(min_value=min_features, max_value=max_features))
    
    # Generate feature data with reasonable variance
    # Use normal distribution to ensure some variance exists
    data = draw(
        arrays(
            dtype=np.float64,
            shape=(n_samples, n_features),
            elements=st.floats(
                min_value=-1000.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False
            )
        )
    )
    
    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return data, feature_names


class TestPCAProperties:
    """Property-based tests for PCA Engine."""
    
    # Feature: llm-customer-segmentation-ads, Property 9a: PCA Variance Threshold
    @given(
        data_and_names=customer_feature_data(min_samples=10, max_samples=50),
        variance_threshold=st.floats(min_value=0.5, max_value=0.99)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_variance_threshold_property(self, data_and_names, variance_threshold):
        """
        **Validates: Requirements 2.1**
        
        Property 9a: PCA Variance Threshold
        
        For any PCA transformation, the number of principal components retained 
        must explain at least 80% of the total variance in the customer feature data.
        
        This test verifies that the PCA engine always retains enough components
        to meet the specified variance threshold.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance (edge case)
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        
        try:
            result = engine.fit_transform(data, feature_names, variance_threshold)
            
            # Property: Cumulative explained variance must meet or exceed threshold
            cumulative_variance = sum(result.explained_variance_ratio)
            
            # Allow small floating point tolerance
            assert cumulative_variance >= variance_threshold - 1e-6, (
                f"PCA did not meet variance threshold. "
                f"Expected >= {variance_threshold}, got {cumulative_variance}"
            )
            
            # Additional invariants
            assert result.n_components >= 1, "Must have at least 1 component"
            assert result.n_components <= len(feature_names), (
                "Cannot have more components than features"
            )
            assert len(result.explained_variance_ratio) == result.n_components
            assert all(0.0 <= ratio <= 1.0 for ratio in result.explained_variance_ratio), (
                "All variance ratios must be between 0 and 1"
            )
            
        except ValueError as e:
            # Some edge cases may raise ValueError (e.g., insufficient samples)
            # This is acceptable behavior
            if "Need at least 2 samples" not in str(e):
                raise
    
    @given(
        data_and_names=customer_feature_data(min_samples=20, max_samples=100)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_default_threshold_80_percent(self, data_and_names):
        """
        **Validates: Requirements 2.1**
        
        Test that default variance threshold of 0.8 (80%) is enforced.
        
        This is the specific requirement from the design document.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names)  # Default threshold = 0.8
        
        # Property: Must explain at least 80% of variance
        cumulative_variance = sum(result.explained_variance_ratio)
        assert cumulative_variance >= 0.8 - 1e-6, (
            f"Default PCA must explain at least 80% variance. Got {cumulative_variance}"
        )
    
    @given(
        data_and_names=customer_feature_data(min_samples=10, max_samples=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_components_orthogonal(self, data_and_names):
        """
        Test that PCA components are orthogonal (mathematical property).
        
        This is a fundamental property of PCA that should always hold.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Convert components to numpy array
        components = np.array(result.components)
        
        # Check orthogonality: dot product of different components should be ~0
        if result.n_components > 1:
            for i in range(result.n_components):
                for j in range(i + 1, result.n_components):
                    dot_product = np.dot(components[i], components[j])
                    assert abs(dot_product) < 1e-6, (
                        f"Components {i} and {j} are not orthogonal. "
                        f"Dot product: {dot_product}"
                    )
    
    @given(
        data_and_names=customer_feature_data(min_samples=10, max_samples=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_explained_variance_descending(self, data_and_names):
        """
        Test that explained variance is in descending order.
        
        This is a fundamental property of PCA - components are ordered by
        the amount of variance they explain.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Property: Explained variance should be in descending order
        for i in range(len(result.explained_variance) - 1):
            assert result.explained_variance[i] >= result.explained_variance[i + 1] - 1e-10, (
                f"Explained variance not in descending order at index {i}"
            )
        
        # Same for explained variance ratio
        for i in range(len(result.explained_variance_ratio) - 1):
            assert result.explained_variance_ratio[i] >= result.explained_variance_ratio[i + 1] - 1e-10, (
                f"Explained variance ratio not in descending order at index {i}"
            )
    
    @given(
        data_and_names=customer_feature_data(min_samples=10, max_samples=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_transform_preserves_samples(self, data_and_names):
        """
        Test that PCA transformation preserves the number of samples.
        
        Dimensionality reduction should reduce features, not samples.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Property: Number of samples should be preserved
        assert len(result.transformed_data) == data.shape[0], (
            f"Number of samples changed. Expected {data.shape[0]}, "
            f"got {len(result.transformed_data)}"
        )
        
        # Each transformed sample should have n_components dimensions
        for sample in result.transformed_data:
            assert len(sample) == result.n_components, (
                f"Transformed sample has wrong dimensionality. "
                f"Expected {result.n_components}, got {len(sample)}"
            )
    
    @given(
        data_and_names=customer_feature_data(min_samples=15, max_samples=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_transform_consistency(self, data_and_names):
        """
        Test that transforming the same data twice produces the same result.
        
        PCA should be deterministic.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        result1 = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Transform the same data again using the fitted model
        transformed2 = engine.transform(data)
        
        # Property: Results should be identical (within floating point tolerance)
        transformed1 = np.array(result1.transformed_data)
        assert np.allclose(transformed1, transformed2, rtol=1e-10, atol=1e-10), (
            "PCA transformation is not consistent"
        )
    
    @given(
        data_and_names=customer_feature_data(min_samples=10, max_samples=50)
    )
    @settings(max_examples=100, deadline=None)
    def test_pca_feature_importance_completeness(self, data_and_names):
        """
        Test that feature importance includes all features.
        
        Every feature should have a loading for each component.
        """
        data, feature_names = data_and_names
        
        # Skip if data has zero variance
        if np.allclose(data, data[0]):
            return
        
        engine = PCAEngine()
        result = engine.fit_transform(data, feature_names, variance_threshold=0.8)
        
        # Property: Feature importance should include all features
        for component_idx in range(result.n_components):
            importance = engine.get_feature_importance(component_idx)
            
            # Should have one entry per feature
            assert len(importance) == len(feature_names), (
                f"Feature importance for component {component_idx} incomplete. "
                f"Expected {len(feature_names)} features, got {len(importance)}"
            )
            
            # All feature names should be present
            importance_names = {name for name, _ in importance}
            assert importance_names == set(feature_names), (
                f"Feature names mismatch in importance for component {component_idx}"
            )
            
            # Should be sorted by absolute value (descending)
            abs_loadings = [abs(loading) for _, loading in importance]
            assert abs_loadings == sorted(abs_loadings, reverse=True), (
                f"Feature importance not sorted for component {component_idx}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
