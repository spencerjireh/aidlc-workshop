"""K-Means Engine for customer clustering."""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.models.ml import ClusteringResult, ClusterStatistics
from src.models.customer import CustomerProfile


class KMeansEngine:
    """Engine for performing K-Means clustering on customer data."""
    
    def __init__(self):
        """Initialize K-Means Engine."""
        self.kmeans = None
        self._is_fitted = False
    
    def determine_optimal_k(
        self,
        pca_data: np.ndarray,
        k_range: Tuple[int, int] = (3, 10)
    ) -> int:
        """
        Determine optimal number of clusters using Elbow Method and Silhouette Score.
        
        Tests k values in range and selects best based on combined metrics.
        
        Args:
            pca_data: PCA-transformed customer data (n_samples, n_components)
            k_range: Tuple of (min_k, max_k) to test (default: 3-10)
        
        Returns:
            Optimal k value
        
        Raises:
            ValueError: If pca_data has insufficient samples or invalid k_range
        """
        min_k, max_k = k_range
        
        if min_k < 2:
            raise ValueError("min_k must be at least 2")
        
        if max_k < min_k:
            raise ValueError("max_k must be greater than or equal to min_k")
        
        n_samples = pca_data.shape[0]
        
        if n_samples < min_k:
            raise ValueError(f"Need at least {min_k} samples for clustering")
        
        # Adjust max_k if we don't have enough samples
        max_k = min(max_k, n_samples - 1)
        
        if max_k < min_k:
            return min_k
        
        # Calculate metrics for each k
        inertias = []
        silhouette_scores = []
        k_values = range(min_k, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pca_data)
            
            inertias.append(kmeans.inertia_)
            
            # Silhouette score requires at least 2 distinct clusters
            n_unique_labels = len(np.unique(labels))
            if n_unique_labels >= 2 and k < n_samples:
                score = silhouette_score(pca_data, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)
        
        # Find optimal k using silhouette score (higher is better)
        optimal_k_idx = np.argmax(silhouette_scores)
        optimal_k = list(k_values)[optimal_k_idx]
        
        return optimal_k
    
    def fit_predict(
        self,
        pca_data: np.ndarray,
        n_clusters: int
    ) -> ClusteringResult:
        """
        Perform K-Means clustering on PCA-transformed data.
        
        Args:
            pca_data: PCA-transformed customer data (n_samples, n_components)
            n_clusters: Number of clusters to create
        
        Returns:
            ClusteringResult with cluster labels, centroids, and metrics
        
        Raises:
            ValueError: If n_clusters is invalid or pca_data has insufficient samples
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        
        n_samples = pca_data.shape[0]
        
        if n_samples < n_clusters:
            raise ValueError(f"Need at least {n_clusters} samples for {n_clusters} clusters")
        
        # Fit K-Means with fixed random_state for determinism
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(pca_data)
        
        self._is_fitted = True
        
        # Calculate silhouette score
        n_unique_labels = len(np.unique(cluster_labels))
        if n_unique_labels >= 2 and n_samples > n_clusters:
            sil_score = silhouette_score(pca_data, cluster_labels)
        else:
            sil_score = 0.0
        
        # Create ClusteringResult
        return ClusteringResult(
            cluster_labels=cluster_labels.tolist(),
            centroids=self.kmeans.cluster_centers_.tolist(),
            inertia=float(self.kmeans.inertia_),
            n_clusters=n_clusters,
            silhouette_score=float(sil_score)
        )
    
    def calculate_confidence_score(
        self,
        customer_point: np.ndarray,
        assigned_cluster: int
    ) -> float:
        """
        Calculate confidence score based on distance to cluster centroid.
        
        Normalized to range [0.0, 1.0] where 1.0 is at centroid.
        Uses a simple inverse relationship: confidence = 1 / (1 + distance)
        
        Args:
            customer_point: Customer data point in PCA space (n_components,)
            assigned_cluster: Cluster ID assigned to customer
        
        Returns:
            Confidence score in range [0.0, 1.0]
        
        Raises:
            ValueError: If K-Means is not fitted or assigned_cluster is invalid
        """
        if not self._is_fitted or self.kmeans is None:
            raise ValueError("K-Means must be fitted before calculating confidence scores")
        
        if assigned_cluster < 0 or assigned_cluster >= self.kmeans.n_clusters:
            raise ValueError(
                f"assigned_cluster must be between 0 and {self.kmeans.n_clusters - 1}"
            )
        
        # Get centroid for assigned cluster
        centroid = self.kmeans.cluster_centers_[assigned_cluster]
        
        # Calculate Euclidean distance to centroid
        distance = np.linalg.norm(customer_point - centroid)
        
        # Use inverse relationship: confidence = 1 / (1 + distance)
        # This guarantees monotonic inverse relationship
        # At distance=0 (at centroid): confidence = 1.0
        # As distance increases: confidence decreases monotonically
        confidence = 1.0 / (1.0 + distance)
        
        # Confidence is already in [0.0, 1.0] by construction
        return float(confidence)
    
    def get_cluster_statistics(
        self,
        cluster_id: int,
        customer_data: List[CustomerProfile]
    ) -> ClusterStatistics:
        """
        Calculate statistics for customers in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            customer_data: List of CustomerProfile objects in this cluster
        
        Returns:
            ClusterStatistics with size, averages, and distributions
        
        Raises:
            ValueError: If customer_data is empty
        """
        if not customer_data:
            raise ValueError("customer_data cannot be empty")
        
        cluster_size = len(customer_data)
        
        # Calculate average age
        average_age = sum(c.age for c in customer_data) / cluster_size
        
        # Calculate location distribution
        location_distribution: Dict[str, int] = {}
        for customer in customer_data:
            location_distribution[customer.location] = \
                location_distribution.get(customer.location, 0) + 1
        
        # Calculate average transaction frequency
        average_transaction_frequency = \
            sum(c.transaction_frequency for c in customer_data) / cluster_size
        
        # Calculate average transaction value
        average_transaction_value = \
            sum(c.average_transaction_value for c in customer_data) / cluster_size
        
        # Calculate total spend distribution (percentiles)
        total_spends = [c.total_spend for c in customer_data]
        total_spend_distribution = {
            'p25': float(np.percentile(total_spends, 25)),
            'p50': float(np.percentile(total_spends, 50)),
            'p75': float(np.percentile(total_spends, 75)),
            'p90': float(np.percentile(total_spends, 90))
        }
        
        # Calculate top merchant categories
        category_counts: Dict[str, int] = {}
        for customer in customer_data:
            for category in customer.merchant_categories:
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Sort by count and get top categories
        top_merchant_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 categories
        
        # Calculate preferred payment methods
        payment_method_counts: Dict[str, int] = {}
        for customer in customer_data:
            for method in customer.preferred_payment_methods:
                payment_method_counts[method] = \
                    payment_method_counts.get(method, 0) + 1
        
        return ClusterStatistics(
            cluster_id=cluster_id,
            size=cluster_size,
            average_age=average_age,
            location_distribution=location_distribution,
            average_transaction_frequency=average_transaction_frequency,
            average_transaction_value=average_transaction_value,
            total_spend_distribution=total_spend_distribution,
            top_merchant_categories=top_merchant_categories,
            preferred_payment_methods=payment_method_counts
        )
