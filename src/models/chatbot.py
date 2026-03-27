"""Chatbot-related data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Message role in conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"


class QueryType(str, Enum):
    """Type of chatbot query."""
    
    SEGMENT_INFO = "segment_info"           # "Tell me about segment X"
    COMPARISON = "comparison"               # "Compare segment A and B"
    PERFORMANCE = "performance"             # "How is campaign X performing?"
    TREND = "trend"                         # "Show trends over time"
    CUSTOMER_COUNT = "customer_count"       # "How many customers in segment X?"
    TOP_CATEGORIES = "top_categories"       # "What are top categories for segment X?"


class QueryIntent(BaseModel):
    """Structured representation of user query intent."""
    
    intent_type: QueryType = Field(..., description="Type of query")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities (segment names, dates, metrics)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent classification confidence")


class ChatMessage(BaseModel):
    """Single message in a conversation."""
    
    message_id: str = Field(..., description="Unique message identifier")
    role: MessageRole = Field(..., description="USER or ASSISTANT")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    query_intent: Optional[QueryIntent] = Field(None, description="Parsed intent for user messages")
    data_references: List[str] = Field(default_factory=list, description="IDs of data used in response")
    
    @field_validator('content')
    @classmethod
    def validate_non_empty_content(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class ConversationContext(BaseModel):
    """Context for a chatbot conversation session."""
    
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Message history")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    last_interaction: datetime = Field(default_factory=datetime.utcnow, description="Last interaction timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")


class ChatbotResponse(BaseModel):
    """Response from the chatbot."""
    
    response_id: str = Field(..., description="Unique response identifier")
    session_id: str = Field(..., description="Session identifier")
    text: str = Field(..., description="Natural language response")
    data: Optional[Dict[str, Any]] = Field(None, description="Supporting data for visualizations")
    visualization_type: Optional[str] = Field(None, description="table, chart, list")
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    response_time_ms: int = Field(..., ge=0, description="Response time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    @field_validator('text')
    @classmethod
    def validate_non_empty_text(cls, v: str) -> str:
        """Ensure response text is not empty."""
        if not v or not v.strip():
            raise ValueError("Response text cannot be empty")
        return v
    
    @field_validator('visualization_type')
    @classmethod
    def validate_visualization_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate visualization type if provided."""
        if v is not None:
            allowed_types = {'table', 'chart', 'list', 'bar', 'pie', 'line'}
            if v not in allowed_types:
                raise ValueError(f"visualization_type must be one of {allowed_types}")
        return v
