"""Unit tests for error handling framework (Task 17.2)."""

import pytest
from unittest.mock import MagicMock

from src.errors import (
    DataValidationError,
    LLMProviderError,
    BusinessLogicError,
    AppSystemError,
    ErrorResponse,
    format_error_response,
    graceful_llm_fallback,
)


class TestExceptionClasses:
    def test_data_validation_error(self):
        err = DataValidationError("cust_123", "age", "must be positive")
        assert err.record_id == "cust_123"
        assert err.field == "age"
        assert err.reason == "must be positive"
        assert "cust_123" in str(err)

    def test_llm_provider_error(self):
        err = LLMProviderError("openai", "429", "Rate limit exceeded")
        assert err.provider == "openai"
        assert err.error_code == "429"
        assert "429" in str(err)

    def test_business_logic_error(self):
        err = BusinessLogicError("min_segment_size", "Segment has only 50 customers")
        assert err.constraint == "min_segment_size"
        assert "min_segment_size" in str(err)

    def test_app_system_error(self):
        err = AppSystemError("database", "Connection refused")
        assert err.component == "database"
        assert "database" in str(err)


class TestErrorResponseFormatting:
    def test_format_validation_error(self):
        err = DataValidationError("cust_123", "age", "must be positive")
        response = format_error_response(err)
        assert response.error.code == "VALIDATION_ERROR"
        assert response.error.details["record_id"] == "cust_123"
        assert response.error.details["field"] == "age"

    def test_format_llm_provider_error(self):
        err = LLMProviderError("openai", "500", "Server error")
        response = format_error_response(err)
        assert response.error.code == "LLM_PROVIDER_ERROR"
        assert response.error.details["provider"] == "openai"

    def test_format_business_logic_error(self):
        err = BusinessLogicError("min_segment_size", "Too few customers")
        response = format_error_response(err)
        assert response.error.code == "BUSINESS_LOGIC_ERROR"
        assert response.error.details["constraint"] == "min_segment_size"

    def test_format_system_error(self):
        err = AppSystemError("database", "Timeout")
        response = format_error_response(err)
        assert response.error.code == "SYSTEM_ERROR"
        assert response.error.details["component"] == "database"

    def test_format_generic_exception(self):
        err = RuntimeError("Something went wrong")
        response = format_error_response(err)
        assert response.error.code == "INTERNAL_ERROR"

    def test_custom_request_id(self):
        err = DataValidationError("c1", "age", "bad")
        response = format_error_response(err, request_id="req_abc123")
        assert response.error.request_id == "req_abc123"

    def test_auto_generated_request_id(self):
        err = DataValidationError("c1", "age", "bad")
        response = format_error_response(err)
        assert len(response.error.request_id) > 0

    def test_timestamp_present(self):
        err = DataValidationError("c1", "age", "bad")
        response = format_error_response(err)
        assert response.error.timestamp is not None


class TestGracefulDegradation:
    def test_successful_operation(self):
        def operation():
            return "success"

        result = graceful_llm_fallback(operation, "fallback")
        assert result == "success"

    def test_llm_failure_returns_fallback(self):
        def operation():
            raise LLMProviderError("openai", "500", "Server error")

        result = graceful_llm_fallback(operation, "fallback_value")
        assert result == "fallback_value"

    def test_non_llm_error_propagates(self):
        def operation():
            raise ValueError("Not an LLM error")

        with pytest.raises(ValueError):
            graceful_llm_fallback(operation, "fallback")

    def test_fallback_with_args(self):
        def operation(x, y):
            raise LLMProviderError("anthropic", "503", "Unavailable")

        result = graceful_llm_fallback(operation, 42, 1, 2)
        assert result == 42
