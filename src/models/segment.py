"""Segmentation-related data models."""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np


class Segment(BaseModel):
    """Customer segment created through clustering."""
    
    segment_id: str = Field(..., description="Unique segment identifier")
    name: str = Field(..., description="LLM-generated descriptive name")
    description: str = Field(..., description="Natural language profile summary")
    characteristics: Dict[str, Any] = Field(..., description="Key characteristics (demographics, behaviors)")
    cluster_id: int = Field(..., ge=0, description="K-Means cluster ID")
    centroid: List[float] = Field(..., description="Cluster centroid in PCA space")
    size: int = Field(..., ge=0, description="Number of customers in segment")
    average_transaction_value: float = Field(..., ge=0, description="Average transaction value for segment")
    transaction_frequency: float = Field(..., ge=0, description="Average transaction frequency")
    top_merchant_categories: List[str] = Field(..., description="Top merchant categories for segment")
    differentiating_factors: List[str] = Field(..., description="What makes this segment unique")
    pca_component_contributions: Dict[int, float] = Field(..., description="PC index -> contribution")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Segment creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Segment update timestamp")
    version: int = Field(default=1, ge=1, description="For segment refinement tracking")
    
    @field_validator('name', 'description')
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Ensure strings are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v
    
    @field_validator('centroid')
    @classmethod
    def validate_centroid(cls, v: List[float]) -> List[float]:
        """Ensure centroid is not empty."""
        if not v:
            raise ValueError("Centroid cannot be empty")
        return v
    
    model_config = ConfigDict(
        json_encoders={
            np.ndarray: lambda x: x.tolist()
        }
    )


class ContributingFactor(BaseModel):
    """Factor contributing to customer-segment assignment."""
    
    factor_name: str = Field(..., description="e.g., 'High transaction frequency'")
    importance: float = Field(..., ge=0.0, le=1.0, description="Relative importance from PCA loadings")
    data_point: str = Field(..., description="Specific data reference")
    pca_loading: Optional[float] = Field(None, description="PCA loading value for this feature")


class CustomerSegmentAssignment(BaseModel):
    """Assignment of a customer to a segment."""
    
    assignment_id: str = Field(..., description="Unique assignment identifier")
    customer_id: str = Field(..., description="Customer identifier")
    segment_id: str = Field(..., description="Segment identifier")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score based on distance to centroid")
    distance_to_centroid: float = Field(..., ge=0.0, description="Euclidean distance in PCA space")
    assigned_at: datetime = Field(default_factory=datetime.utcnow, description="Assignment timestamp")
    explanation: Optional[str] = Field(None, description="Natural language explanation")
    contributing_factors: List[ContributingFactor] = Field(default_factory=list, description="Contributing factors")



class SegmentProfile(BaseModel):
    """Profile information for a customer segment (LLM-generated)."""
    
    segment_id: str = Field(..., description="Segment identifier")
    name: str = Field(..., description="LLM-generated descriptive name")
    description: str = Field(..., description="Natural language profile summary")
    differentiating_factors: List[str] = Field(..., description="What makes this segment unique")
    
    @field_validator('name', 'description')
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Ensure strings are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v
