"""Data models for the LLM-powered customer segmentation advertising system."""

# Customer models
from src.models.customer import (
    CustomerProfile,
    TransactionData,
)

# Segment models
from src.models.segment import (
    Segment,
    CustomerSegmentAssignment,
    ContributingFactor,
)

# ML models
from src.models.ml import (
    PCAResult,
    ClusteringResult,
    ClusterStatistics,
)

# Campaign models
from src.models.campaign import (
    AdContent,
    AdFormat,
    Campaign,
    CampaignStatus,
    AdPerformanceMetrics,
)

# Chatbot models
from src.models.chatbot import (
    ConversationContext,
    ChatMessage,
    QueryIntent,
    ChatbotResponse,
    MessageRole,
    QueryType,
)

# LLM models
from src.models.llm import (
    LLMConfiguration,
    LLMProvider,
    LLMParameters,
)

# Service models
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
    # Segment models
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
    "Campaign",
    "CampaignStatus",
    "AdPerformanceMetrics",
    # Chatbot models
    "ConversationContext",
    "ChatMessage",
    "QueryIntent",
    "ChatbotResponse",
    "MessageRole",
    "QueryType",
    # LLM models
    "LLMConfiguration",
    "LLMProvider",
    "LLMParameters",
    # Service models
    "DataFormat",
    "IngestionResult",
    "ContentValidationResult",
    "ReachEstimate",
    "CampaignActivationResult",
]
