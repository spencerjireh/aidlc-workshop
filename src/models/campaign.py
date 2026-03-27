"""Campaign and ad-related data models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class AdFormat(str, Enum):
    """Ad format with character limits."""
    
    SHORT = "short"      # 50 characters
    MEDIUM = "medium"    # 150 characters
    LONG = "long"        # 300 characters


class CampaignStatus(str, Enum):
    """Campaign status values."""
    
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class AdPerformanceMetrics(BaseModel):
    """Performance metrics for an ad."""
    
    ad_id: str = Field(..., description="Ad identifier")
    impressions: int = Field(..., ge=0, description="Number of impressions")
    clicks: int = Field(..., ge=0, description="Number of clicks")
    conversions: int = Field(..., ge=0, description="Number of conversions")
    click_through_rate: float = Field(..., ge=0.0, le=1.0, description="clicks / impressions")
    conversion_rate: float = Field(..., ge=0.0, le=1.0, description="conversions / clicks")
    segment_id: str = Field(..., description="Target segment identifier")
    measurement_period_start: datetime = Field(..., description="Measurement period start")
    measurement_period_end: datetime = Field(..., description="Measurement period end")
    
    @field_validator('clicks')
    @classmethod
    def validate_clicks_not_exceed_impressions(cls, v: int, info) -> int:
        """Ensure clicks don't exceed impressions."""
        if 'impressions' in info.data and v > info.data['impressions']:
            raise ValueError("Clicks cannot exceed impressions")
        return v
    
    @field_validator('conversions')
    @classmethod
    def validate_conversions_not_exceed_clicks(cls, v: int, info) -> int:
        """Ensure conversions don't exceed clicks."""
        if 'clicks' in info.data and v > info.data['clicks']:
            raise ValueError("Conversions cannot exceed clicks")
        return v


class AdContent(BaseModel):
    """Ad content for a campaign."""
    
    ad_id: str = Field(..., description="Unique ad identifier")
    segment_id: str = Field(..., description="Target segment identifier")
    campaign_id: str = Field(..., description="Campaign identifier")
    format: AdFormat = Field(..., description="Ad format with character limit")
    content: str = Field(..., description="The ad copy")
    variation_number: int = Field(..., ge=1, description="For A/B testing (1, 2, 3, etc.)")
    use_case: str = Field(..., description="cashback, merchant_promo, payment_convenience")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    performance_metrics: Optional[AdPerformanceMetrics] = Field(None, description="Performance metrics")
    
    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v: str, info) -> str:
        """Validate content length based on format."""
        if 'format' not in info.data:
            return v
        
        format_limits = {
            AdFormat.SHORT: 50,
            AdFormat.MEDIUM: 150,
            AdFormat.LONG: 300
        }
        
        max_length = format_limits.get(info.data['format'])
        if max_length and len(v) > max_length:
            raise ValueError(f"Content exceeds {max_length} character limit for {info.data['format']} format")
        
        return v
    
    @field_validator('use_case')
    @classmethod
    def validate_use_case(cls, v: str) -> str:
        """Validate use case is one of the allowed values."""
        allowed_use_cases = {'cashback', 'merchant_promo', 'payment_convenience'}
        if v not in allowed_use_cases:
            raise ValueError(f"use_case must be one of {allowed_use_cases}")
        return v


class Campaign(BaseModel):
    """Marketing campaign targeting customer segments."""
    
    campaign_id: str = Field(..., description="Unique campaign identifier")
    name: str = Field(..., description="Campaign name")
    target_segment_ids: List[str] = Field(..., description="Target segment identifiers")
    ad_content_ids: List[str] = Field(..., description="Ad content identifiers")
    status: CampaignStatus = Field(default=CampaignStatus.DRAFT, description="Campaign status")
    estimated_reach: int = Field(..., ge=0, description="Estimated number of customers reached")
    start_date: Optional[datetime] = Field(None, description="Campaign start date")
    end_date: Optional[datetime] = Field(None, description="Campaign end date")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    
    @field_validator('target_segment_ids', 'ad_content_ids')
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        """Ensure lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v
    
    @field_validator('end_date')
    @classmethod
    def validate_end_date_after_start(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure end date is after start date."""
        if v and 'start_date' in info.data and info.data['start_date']:
            if v <= info.data['start_date']:
                raise ValueError("end_date must be after start_date")
        return v
