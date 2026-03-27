"""Configuration management for LLM providers and database connections."""

import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMProviderConfig(BaseModel):
    """Configuration for a specific LLM provider."""
    provider: LLMProvider
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    model_name: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "segmentation_db"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20



class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    app_name: str = "LLM Customer Segmentation Ads"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # LLM Configuration
    default_llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-opus-20240229"
    local_model_endpoint: Optional[str] = None
    local_model_name: str = "llama2"
    
    # LLM Parameters
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    llm_retry_attempts: int = 3
    llm_retry_delay: float = 1.0
    
    # Database
    database_url: str = "sqlite:///./segmentation.db"  # Default to SQLite for POC
    
    # ML Configuration
    pca_variance_threshold: float = 0.8
    kmeans_min_clusters: int = 3
    kmeans_max_clusters: int = 10
    kmeans_random_state: int = 42
    
    # Business Rules
    min_segment_size: int = 100
    min_ad_variations: int = 3
    
    # Performance
    data_ingestion_batch_size: int = 1000
    cache_ttl_seconds: int = 3600


# Global settings instance
settings = Settings()


def get_llm_config(provider: Optional[LLMProvider] = None) -> LLMProviderConfig:
    """Get LLM configuration for the specified provider."""
    if provider is None:
        provider = settings.default_llm_provider
    
    if provider == LLMProvider.OPENAI:
        return LLMProviderConfig(
            provider=LLMProvider.OPENAI,
            api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    elif provider == LLMProvider.ANTHROPIC:
        return LLMProviderConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=settings.anthropic_api_key,
            model_name=settings.anthropic_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    elif provider == LLMProvider.LOCAL:
        return LLMProviderConfig(
            provider=LLMProvider.LOCAL,
            api_endpoint=settings.local_model_endpoint,
            model_name=settings.local_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    # For POC, we'll use SQLite, but this can be extended for PostgreSQL/MongoDB
    return DatabaseConfig(
        host="localhost",
        port=5432,
        database="segmentation_db",
        username="postgres",
        password=os.getenv("DB_PASSWORD", ""),
    )


