"""
Consolidated data models for the LLM-Powered Customer Segmentation Advertising System.

This module provides a single import point for all data models used throughout the system,
including customer profiles, segmentation results, campaigns, chatbot interactions, and LLM configurations.

All models use Pydantic v2 for validation and type checking.
"""

# Customer-related models
from src.models.customer import (
    CustomerProfile,
    TransactionData,
)

# Segmentation-related models
from src.models.segment import (
    Segment,
    CustomerSegmentAssignment,
    ContributingFactor,
)

# ML-related models
from src.models.ml import (
    PCAResult,
    ClusteringResult,
    ClusterStatistics,
)

# Campaign and ad-related models
from src.models.campaign import (
    AdContent,
    AdFormat,
    AdPerformanceMetrics,
    Campaign,
    CampaignStatus,
)

# LLM configuration models
from src.models.llm import (
    LLMConfiguration,
    LLMParameters,
    LLMProvider,
)

# Chatbot-related models
from src.models.chatbot import (
    ConversationContext,
    ChatMessage,
    MessageRole,
    QueryIntent,
    QueryType,
    ChatbotResponse,
)

# Service-layer models
from src.models.service_models import (
    DataFormat,
    IngestionResult,
    ContentValidationResult,
    ReachEstimate,
    CampaignActivationResult,
)

__all__ = [
    # Customer models
    "CustomerProfile",
    "TransactionData",
    # Segmentation models
    "Segment",
    "CustomerSegmentAssignment",
    "ContributingFactor",
    # ML models
    "PCAResult",
    "ClusteringResult",
    "ClusterStatistics",
    # Campaign models
    "AdContent",
    "AdFormat",
    "AdPerformanceMetrics",
    "Campaign",
    "CampaignStatus",
    # LLM models
    "LLMConfiguration",
    "LLMParameters",
    "LLMProvider",
    # Chatbot models
    "ConversationContext",
    "ChatMessage",
    "MessageRole",
    "QueryIntent",
    "QueryType",
    "ChatbotResponse",
    # Service models
    "DataFormat",
    "IngestionResult",
    "ContentValidationResult",
    "ReachEstimate",
    "CampaignActivationResult",
]
