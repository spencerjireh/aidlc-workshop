"""ML-related data models for PCA and clustering."""

from typing import Dict, List, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np


class PCAResult(BaseModel):
    """Result of Principal Component Analysis."""
    
    transformed_data: List[List[float]] = Field(..., description="Customer data in PCA space")
    explained_variance: List[float] = Field(..., description="Variance explained by each component")
    explained_variance_ratio: List[float] = Field(..., description="Ratio of variance explained")
    components: List[List[float]] = Field(..., description="Principal component vectors")
    feature_names: List[str] = Field(..., description="Original feature names")
    n_components: int = Field(..., gt=0, description="Number of components retained")
    
    @field_validator('explained_variance_ratio')
    @classmethod
    def validate_variance_ratio(cls, v: List[float]) -> List[float]:
        """Ensure variance ratios are between 0 and 1."""
        for ratio in v:
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("Variance ratio must be between 0 and 1")
        return v
    
    model_config = ConfigDict(
        json_encoders={
            np.ndarray: lambda x: x.tolist()
        }
    )


class ClusteringResult(BaseModel):
    """Result of K-Means clustering."""
    
    cluster_labels: List[int] = Field(..., description="Cluster assignment for each customer")
    centroids: List[List[float]] = Field(..., description="Cluster centroids in PCA space")
    inertia: float = Field(..., ge=0.0, description="Sum of squared distances to centroids")
    n_clusters: int = Field(..., ge=1, description="Number of clusters")
    silhouette_score: float = Field(..., ge=-1.0, le=1.0, description="Clustering quality metric")
    
    @field_validator('cluster_labels')
    @classmethod
    def validate_cluster_labels(cls, v: List[int]) -> List[int]:
        """Ensure cluster labels are non-negative."""
        if any(label < 0 for label in v):
            raise ValueError("Cluster labels must be non-negative")
        return v
    
    model_config = ConfigDict(
        json_encoders={
            np.ndarray: lambda x: x.tolist()
        }
    )


class ClusterStatistics(BaseModel):
    """Statistics for a specific cluster."""
    
    cluster_id: int = Field(..., ge=0, description="Cluster identifier")
    size: int = Field(..., ge=0, description="Number of customers in cluster")
    average_age: float = Field(..., ge=0.0, description="Average age of customers")
    location_distribution: Dict[str, int] = Field(..., description="Location -> count mapping")
    average_transaction_frequency: float = Field(..., ge=0.0, description="Average transaction frequency")
    average_transaction_value: float = Field(..., ge=0.0, description="Average transaction value")
    total_spend_distribution: Dict[str, float] = Field(..., description="Percentiles of total spend")
    top_merchant_categories: List[Tuple[str, int]] = Field(..., description="(category, count) tuples")
    preferred_payment_methods: Dict[str, int] = Field(..., description="Payment method -> count mapping")
    
    @field_validator('top_merchant_categories')
    @classmethod
    def validate_merchant_categories(cls, v: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Ensure merchant categories have valid counts."""
        for category, count in v:
            if count < 0:
                raise ValueError("Category count must be non-negative")
        return v
