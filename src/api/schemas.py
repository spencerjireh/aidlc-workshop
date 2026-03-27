"""Request and response schemas for API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.models.campaign import AdFormat
from src.models.llm import LLMProvider


class CustomerIngestRequest(BaseModel):
    customers: List[Dict[str, Any]]
    format: str = "json"


class SegmentCreateRequest(BaseModel):
    customer_ids: Optional[List[str]] = None
    num_clusters: Optional[int] = Field(None, ge=3, le=10)


class SegmentRefineRequest(BaseModel):
    num_clusters: int = Field(..., ge=3, le=10)


class AdGenerateRequest(BaseModel):
    segment_id: str
    campaign_brief: str
    formats: List[AdFormat] = Field(default_factory=lambda: [AdFormat.SHORT, AdFormat.MEDIUM, AdFormat.LONG])
    num_variations: int = Field(default=3, ge=1)


class CampaignCreateRequest(BaseModel):
    name: str
    target_segment_ids: List[str]
    ad_content_ids: List[str]


class ChatbotQueryRequest(BaseModel):
    query: str
    session_id: str
    user_id: str = "anonymous"


class LLMConfigureRequest(BaseModel):
    provider: LLMProvider
    model_name: str
    api_key: str
    api_endpoint: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)


class LLMValidateRequest(BaseModel):
    provider: LLMProvider
    api_key: str
    model_name: str = "gpt-4"
    api_endpoint: Optional[str] = None
