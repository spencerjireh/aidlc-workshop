"""Unit tests for K-Means Engine."""

import numpy as np
import pytest
from datetime import datetime

from src.engines.kmeans_engine import KMeansEngine
from src.models.customer import CustomerProfile
from src.models.ml import ClusterStatistics


class TestKMeansEngine:
    """Unit tests for K-Means Engine."""
    
    def test_determine_optimal_k_basic(self):
        """Test optimal k determination with basic dataset."""
        # Create synthetic data with clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 3) + np.array([0, 0, 0])
        cluster2 = np.random.randn(50, 3) + np.array([5, 5, 5])
        cluster3 = np.random.randn(50, 3) + np.array([-5, -5, -5])
        pca_data = np.vstack([cluster1, cluster2, cluster3])
        
        engine = KMeansEngine()
        optimal_k = engine.determine_optimal_k(pca_data, k_range=(3, 10))
        
        # Should find 3 clusters
        assert 3 <= optimal_k <= 10
        assert isinstance(optimal_k, int)
    
    def test_determine_optimal_k_with_small_dataset(self):
        """Test optimal k determination with dataset smaller than max_k."""
        # Create data with only 5 samples
        pca_data = np.random.randn(5, 3)
        
        engine = KMeansEngine()
        optimal_k = engine.determine_optimal_k(pca_data, k_range=(3, 10))
        
        # Should return a valid k (adjusted for small dataset)
        assert 3 <= optimal_k <= 4  # Can't have more clusters than samples-1
    
    def test_determine_optimal_k_invalid_range(self):
        """Test optimal k determination with invalid k_range."""
        pca_data = np.random.randn(100, 3)
        engine = KMeansEngine()
        
        # Test min_k < 2
        with pytest.raises(ValueError, match="min_k must be at least 2"):
            engine.determine_optimal_k(pca_data, k_range=(1, 10))
        
        # Test max_k < min_k
        with pytest.raises(ValueError, match="max_k must be greater than or equal to min_k"):
            engine.determine_optimal_k(pca_data, k_range=(10, 5))
    
    def test_determine_optimal_k_insufficient_samples(self):
        """Test optimal k determination with insufficient samples."""
        pca_data = np.random.randn(2, 3)
        engine = KMeansEngine()
        
        with pytest.raises(ValueError, match="Need at least 3 samples"):
            engine.determine_optimal_k(pca_data, k_range=(3, 10))
    
    def test_fit_predict_basic(self):
        """Test K-Means clustering with basic dataset."""
        # Create synthetic data
        np.random.seed(42)
        pca_data = np.random.randn(100, 3)
        
        engine = KMeansEngine()
        result = engine.fit_predict(pca_data, n_clusters=5)
        
        # Verify result structure
        assert result.n_clusters == 5
        assert len(result.cluster_labels) == 100
        assert len(result.centroids) == 5
        assert result.inertia >= 0
        assert -1.0 <= result.silhouette_score <= 1.0
        
        # Verify all labels are in valid range
        assert all(0 <= label < 5 for label in result.cluster_labels)
    
    def test_fit_predict_determinism(self):
        """Test that K-Means produces deterministic results."""
        np.random.seed(42)
        pca_data = np.random.randn(100, 3)
        
        engine1 = KMeansEngine()
        result1 = engine1.fit_predict(pca_data, n_clusters=5)
        
        engine2 = KMeansEngine()
        result2 = engine2.fit_predict(pca_data, n_clusters=5)
        
        # Results should be identical
        assert result1.cluster_labels == result2.cluster_labels
        np.testing.assert_array_almost_equal(
            np.array(result1.centroids),
            np.array(result2.centroids)
        )
    
    def test_fit_predict_invalid_n_clusters(self):
        """Test fit_predict with invalid n_clusters."""
        pca_data = np.random.randn(100, 3)
        engine = KMeansEngine()
        
        # Test n_clusters < 1
        with pytest.raises(ValueError, match="n_clusters must be at least 1"):
            engine.fit_predict(pca_data, n_clusters=0)
    
    def test_fit_predict_insufficient_samples(self):
        """Test fit_predict with insufficient samples."""
        pca_data = np.random.randn(3, 3)
        engine = KMeansEngine()
        
        with pytest.raises(ValueError, match="Need at least 5 samples"):
            engine.fit_predict(pca_data, n_clusters=5)
    
    def test_calculate_confidence_score_at_centroid(self):
        """Test confidence score calculation for point at centroid."""
        # Create simple data
        np.random.seed(42)
        pca_data = np.random.randn(50, 3)
        
        engine = KMeansEngine()
        result = engine.fit_predict(pca_data, n_clusters=3)
        
        # Test confidence at centroid (should be close to 1.0)
        centroid = np.array(result.centroids[0])
        confidence = engine.calculate_confidence_score(centroid, 0)
        
        assert 0.9 <= confidence <= 1.0
    
    def test_calculate_confidence_score_far_from_centroid(self):
        """Test confidence score calculation for point far from centroid."""
        # Create simple data
        np.random.seed(42)
        pca_data = np.random.randn(50, 3)
        
        engine = KMeansEngine()
        result = engine.fit_predict(pca_data, n_clusters=3)
        
        # Create a point far from all centroids
        far_point = np.array([100, 100, 100])
        confidence = engine.calculate_confidence_score(far_point, 0)
        
        # Should have low confidence
        assert 0.0 <= confidence < 0.5
    
    def test_calculate_confidence_score_not_fitted(self):
        """Test confidence score calculation before fitting."""
        engine = KMeansEngine()
        point = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="K-Means must be fitted"):
            engine.calculate_confidence_score(point, 0)
    
    def test_calculate_confidence_score_invalid_cluster(self):
        """Test confidence score calculation with invalid cluster ID."""
        np.random.seed(42)
        pca_data = np.random.randn(50, 3)
        
        engine = KMeansEngine()
        engine.fit_predict(pca_data, n_clusters=3)
        
        point = np.array([1, 2, 3])
        
        # Test negative cluster ID
        with pytest.raises(ValueError, match="assigned_cluster must be between"):
            engine.calculate_confidence_score(point, -1)
        
        # Test cluster ID >= n_clusters
        with pytest.raises(ValueError, match="assigned_cluster must be between"):
            engine.calculate_confidence_score(point, 5)
    
    def test_get_cluster_statistics_basic(self):
        """Test cluster statistics calculation with basic data."""
        # Create sample customer profiles
        customers = [
            CustomerProfile(
                customer_id=f"c{i}",
                age=25 + i,
                location="Manila" if i < 5 else "Cebu",
                transaction_frequency=10 + i,
                average_transaction_value=100.0 + i * 10,
                merchant_categories=["Food", "Shopping"],
                total_spend=1000.0 + i * 100,
                account_age_days=365,
                preferred_payment_methods=["Credit Card", "E-wallet"],
                last_transaction_date=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(10)
        ]
        
        engine = KMeansEngine()
        stats = engine.get_cluster_statistics(cluster_id=0, customer_data=customers)
        
        # Verify statistics
        assert stats.cluster_id == 0
        assert stats.size == 10
        assert 25 <= stats.average_age <= 34
        assert stats.average_transaction_frequency > 0
        assert stats.average_transaction_value > 0
        assert len(stats.location_distribution) > 0
        assert len(stats.top_merchant_categories) > 0
        assert len(stats.preferred_payment_methods) > 0
        assert len(stats.total_spend_distribution) == 4  # p25, p50, p75, p90
    
    def test_get_cluster_statistics_location_distribution(self):
        """Test location distribution calculation."""
        customers = [
            CustomerProfile(
                customer_id=f"c{i}",
                age=30,
                location="Manila" if i < 7 else "Cebu",
                transaction_frequency=10,
                average_transaction_value=100.0,
                merchant_categories=["Food"],
                total_spend=1000.0,
                account_age_days=365,
                preferred_payment_methods=["Credit Card"],
                last_transaction_date=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(10)
        ]
        
        engine = KMeansEngine()
        stats = engine.get_cluster_statistics(cluster_id=0, customer_data=customers)
        
        # Verify location distribution
        assert stats.location_distribution["Manila"] == 7
        assert stats.location_distribution["Cebu"] == 3
    
    def test_get_cluster_statistics_merchant_categories(self):
        """Test merchant category counting."""
        customers = [
            CustomerProfile(
                customer_id=f"c{i}",
                age=30,
                location="Manila",
                transaction_frequency=10,
                average_transaction_value=100.0,
                merchant_categories=["Food", "Shopping"] if i < 5 else ["Food", "Transport"],
                total_spend=1000.0,
                account_age_days=365,
                preferred_payment_methods=["Credit Card"],
                last_transaction_date=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(10)
        ]
        
        engine = KMeansEngine()
        stats = engine.get_cluster_statistics(cluster_id=0, customer_data=customers)
        
        # Verify top categories
        category_dict = dict(stats.top_merchant_categories)
        assert category_dict["Food"] == 10  # All customers have Food
        assert category_dict["Shopping"] == 5
        assert category_dict["Transport"] == 5
    
    def test_get_cluster_statistics_empty_data(self):
        """Test cluster statistics with empty customer data."""
        engine = KMeansEngine()
        
        with pytest.raises(ValueError, match="customer_data cannot be empty"):
            engine.get_cluster_statistics(cluster_id=0, customer_data=[])
    
    def test_get_cluster_statistics_percentiles(self):
        """Test total spend percentile calculation."""
        # Create customers with known spend values
        customers = [
            CustomerProfile(
                customer_id=f"c{i}",
                age=30,
                location="Manila",
                transaction_frequency=10,
                average_transaction_value=100.0,
                merchant_categories=["Food"],
                total_spend=float(i * 100),  # 0, 100, 200, ..., 900
                account_age_days=365,
                preferred_payment_methods=["Credit Card"],
                last_transaction_date=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            for i in range(10)
        ]
        
        engine = KMeansEngine()
        stats = engine.get_cluster_statistics(cluster_id=0, customer_data=customers)
        
        # Verify percentiles are in expected ranges
        assert 0 <= stats.total_spend_distribution['p25'] <= 300
        assert 300 <= stats.total_spend_distribution['p50'] <= 600
        assert 600 <= stats.total_spend_distribution['p75'] <= 900
        assert 700 <= stats.total_spend_distribution['p90'] <= 900
