"""Property-based tests for K-Means Engine."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from src.engines.kmeans_engine import KMeansEngine
from src.models.customer import CustomerProfile


# Strategy for generating valid PCA data
def pca_data_strategy(min_samples=10, max_samples=1000, min_components=2, max_components=10):
    """Generate valid PCA-transformed data with some variance."""
    return st.integers(min_value=min_samples, max_value=max_samples).flatmap(
        lambda n_samples: st.integers(min_value=min_components, max_value=max_components).flatmap(
            lambda n_components: arrays(
                dtype=np.float64,
                shape=(n_samples, n_components),
                elements=st.floats(
                    min_value=-10.0, 
                    max_value=10.0, 
                    allow_nan=False, 
                    allow_infinity=False
                )
            ).filter(lambda arr: np.std(arr) > 0.01)  # Filter out arrays with no variance
        )
    )


class TestKMeansProperties:
    """Property-based tests for K-Means Engine."""
    
    @given(pca_data=pca_data_strategy(min_samples=100, max_samples=500))
    @settings(max_examples=100, deadline=None)
    def test_segment_count_bounds(self, pca_data):
        """
        **Validates: Requirements 2.2, 2.3**
        
        Property 4: Segment Count Bounds
        For any customer dataset, the number of segments created using K-Means 
        should be between 3 and 10 (inclusive).
        """
        engine = KMeansEngine()
        
        # Determine optimal k
        optimal_k = engine.determine_optimal_k(pca_data, k_range=(3, 10))
        
        # Verify k is in valid range
        assert 3 <= optimal_k <= 10, f"Optimal k {optimal_k} is outside range [3, 10]"
        
        # Verify clustering with optimal k works
        result = engine.fit_predict(pca_data, optimal_k)
        assert result.n_clusters == optimal_k
        assert 3 <= result.n_clusters <= 10
    
    @given(
        pca_data=pca_data_strategy(min_samples=50, max_samples=200),
        n_clusters=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_confidence_score_bounds(self, pca_data, n_clusters):
        """
        **Validates: Requirements 2.6**
        
        Property 7: Confidence Score Bounds
        For any customer-segment assignment, the confidence score (calculated from 
        normalized distance to cluster centroid) must be in the range [0.0, 1.0] inclusive.
        """
        # Skip if not enough samples
        if pca_data.shape[0] < n_clusters:
            return
        
        engine = KMeansEngine()
        
        # Fit K-Means
        result = engine.fit_predict(pca_data, n_clusters)
        
        # Test confidence scores for all data points
        for i, point in enumerate(pca_data):
            assigned_cluster = result.cluster_labels[i]
            confidence = engine.calculate_confidence_score(point, assigned_cluster)
            
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence score {confidence} is outside range [0.0, 1.0]"
    
    @given(
        pca_data=pca_data_strategy(min_samples=50, max_samples=200),
        n_clusters=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_kmeans_determinism(self, pca_data, n_clusters):
        """
        **Validates: Requirements 2.5**
        
        Property 9b: K-Means Cluster Assignment Determinism
        For any customer and fitted K-Means model, running cluster assignment 
        multiple times on the same customer data must produce the same cluster labels.
        """
        # Skip if not enough samples
        if pca_data.shape[0] < n_clusters:
            return
        
        engine1 = KMeansEngine()
        engine2 = KMeansEngine()
        
        # Fit both engines with same data and parameters
        result1 = engine1.fit_predict(pca_data, n_clusters)
        result2 = engine2.fit_predict(pca_data, n_clusters)
        
        # Cluster labels should be identical (deterministic)
        assert result1.cluster_labels == result2.cluster_labels, \
            "K-Means clustering is not deterministic"
        
        # Centroids should be identical
        np.testing.assert_array_almost_equal(
            np.array(result1.centroids),
            np.array(result2.centroids),
            decimal=10,
            err_msg="K-Means centroids are not deterministic"
        )
        
        # Inertia should be identical
        assert abs(result1.inertia - result2.inertia) < 1e-10, \
            "K-Means inertia is not deterministic"
    
    @given(
        pca_data=pca_data_strategy(min_samples=50, max_samples=200),
        n_clusters=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_confidence_score_inverse_distance_relationship(self, pca_data, n_clusters):
        """
        **Validates: Requirements 2.6**
        
        Property 9c: Distance-Based Confidence Score
        For any customer-segment assignment, the confidence score must be inversely 
        related to the distance to the cluster centroid (customers closer to centroid 
        have higher confidence scores).
        """
        # Skip if not enough samples
        if pca_data.shape[0] < n_clusters:
            return
        
        engine = KMeansEngine()
        result = engine.fit_predict(pca_data, n_clusters)
        
        # For each cluster, verify inverse relationship
        for cluster_id in range(n_clusters):
            # Get all points in this cluster
            cluster_points = [
                (i, pca_data[i]) 
                for i, label in enumerate(result.cluster_labels) 
                if label == cluster_id
            ]
            
            if len(cluster_points) < 2:
                continue
            
            # Calculate distances and confidence scores
            centroid = np.array(result.centroids[cluster_id])
            distances_and_confidences = []
            
            for idx, point in cluster_points:
                distance = np.linalg.norm(point - centroid)
                confidence = engine.calculate_confidence_score(point, cluster_id)
                distances_and_confidences.append((distance, confidence))
            
            # Sort by distance
            distances_and_confidences.sort(key=lambda x: x[0])
            
            # Verify inverse relationship: as distance increases, confidence should not increase
            # Allow small tolerance for floating point errors
            tolerance = 1e-9
            for i in range(len(distances_and_confidences) - 1):
                dist1, conf1 = distances_and_confidences[i]
                dist2, conf2 = distances_and_confidences[i + 1]
                
                # If distance increases significantly, confidence should decrease or stay same
                if dist2 > dist1 + tolerance:
                    assert conf2 <= conf1 + tolerance, \
                        f"Confidence score not inversely related to distance: " \
                        f"dist1={dist1}, conf1={conf1}, dist2={dist2}, conf2={conf2}"
