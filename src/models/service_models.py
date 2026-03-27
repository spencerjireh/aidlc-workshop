"""Service-layer return type models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.models.campaign import CampaignStatus


class DataFormat(str, Enum):
    """Supported data ingestion formats."""
    JSON = "json"
    CSV = "csv"


class IngestionResult(BaseModel):
    """Result of a customer data ingestion operation."""
    total_records: int = Field(..., ge=0)
    successful: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    duplicates_merged: int = Field(..., ge=0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class ContentValidationResult(BaseModel):
    """Result of ad content validation."""
    is_approved: bool
    issues: List[str] = Field(default_factory=list)


class ReachEstimate(BaseModel):
    """Estimated audience reach for a set of segments."""
    total_customers: int = Field(..., ge=0)
    segment_breakdown: Dict[str, int] = Field(default_factory=dict)


class CampaignActivationResult(BaseModel):
    """Result of activating a campaign."""
    campaign_id: str
    status: CampaignStatus
    activated_at: datetime = Field(default_factory=datetime.utcnow)
    segments_targeted: List[str] = Field(default_factory=list)
    ads_associated: List[str] = Field(default_factory=list)
