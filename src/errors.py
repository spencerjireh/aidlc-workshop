"""Custom exception classes and error response formatting."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DataValidationError(Exception):
    """Raised when customer data validation fails."""

    def __init__(self, record_id: str, field: str, reason: str):
        self.record_id = record_id
        self.field = field
        self.reason = reason
        super().__init__(f"Validation failed for {record_id}.{field}: {reason}")


class LLMProviderError(Exception):
    """Raised when an LLM API call fails."""

    def __init__(self, provider: str, error_code: str, message: str):
        self.provider = provider
        self.error_code = error_code
        super().__init__(f"LLM Provider {provider} error [{error_code}]: {message}")


class BusinessLogicError(Exception):
    """Raised when a business constraint is violated."""

    def __init__(self, constraint: str, message: str):
        self.constraint = constraint
        super().__init__(f"Business constraint violated [{constraint}]: {message}")


class AppSystemError(Exception):
    """Raised for system-level failures (database, network)."""

    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(f"System error in {component}: {message}")


class ErrorDetail(BaseModel):
    """Structured error response."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


class ErrorResponse(BaseModel):
    """Top-level error response wrapper."""

    error: ErrorDetail


def format_error_response(
    exception: Exception, request_id: Optional[str] = None
) -> ErrorResponse:
    """Convert an exception into a structured ErrorResponse."""
    detail_kwargs: Dict[str, Any] = {}
    if request_id:
        detail_kwargs["request_id"] = request_id

    if isinstance(exception, DataValidationError):
        return ErrorResponse(
            error=ErrorDetail(
                code="VALIDATION_ERROR",
                message=str(exception),
                details={
                    "record_id": exception.record_id,
                    "field": exception.field,
                    "reason": exception.reason,
                },
                **detail_kwargs,
            )
        )
    elif isinstance(exception, LLMProviderError):
        return ErrorResponse(
            error=ErrorDetail(
                code="LLM_PROVIDER_ERROR",
                message=str(exception),
                details={
                    "provider": exception.provider,
                    "error_code": exception.error_code,
                },
                **detail_kwargs,
            )
        )
    elif isinstance(exception, BusinessLogicError):
        return ErrorResponse(
            error=ErrorDetail(
                code="BUSINESS_LOGIC_ERROR",
                message=str(exception),
                details={"constraint": exception.constraint},
                **detail_kwargs,
            )
        )
    elif isinstance(exception, AppSystemError):
        return ErrorResponse(
            error=ErrorDetail(
                code="SYSTEM_ERROR",
                message=str(exception),
                details={"component": exception.component},
                **detail_kwargs,
            )
        )
    else:
        return ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message=str(exception),
                **detail_kwargs,
            )
        )


def graceful_llm_fallback(operation, fallback_value, *args, **kwargs):
    """
    Wrap an LLM call with graceful degradation.

    If the operation raises LLMProviderError, returns fallback_value instead.
    """
    try:
        return operation(*args, **kwargs)
    except LLMProviderError:
        return fallback_value
