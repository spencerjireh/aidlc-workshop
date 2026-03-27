"""LLM configuration and provider data models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class LLMProvider(str, Enum):
    """LLM provider options."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMParameters(BaseModel):
    """Parameters for LLM generation."""
    
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")


class LLMConfiguration(BaseModel):
    """Configuration for an LLM provider."""
    
    config_id: str = Field(..., description="Unique configuration identifier")
    provider: LLMProvider = Field(..., description="LLM provider (OPENAI, ANTHROPIC, LOCAL)")
    model_name: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3-opus')")
    api_key: str = Field(..., description="API key (should be encrypted)")
    api_endpoint: Optional[str] = Field(None, description="API endpoint for local models")
    parameters: LLMParameters = Field(default_factory=LLMParameters, description="Generation parameters")
    is_active: bool = Field(default=True, description="Whether this configuration is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Configuration creation timestamp")
    
    @field_validator('model_name')
    @classmethod
    def validate_non_empty_model_name(cls, v: str) -> str:
        """Ensure model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v
    
    @field_validator('api_endpoint')
    @classmethod
    def validate_local_endpoint(cls, v: Optional[str], info) -> Optional[str]:
        """Validate that local provider has an endpoint."""
        if 'provider' in info.data and info.data['provider'] == LLMProvider.LOCAL:
            if not v:
                raise ValueError("api_endpoint is required for LOCAL provider")
        return v
