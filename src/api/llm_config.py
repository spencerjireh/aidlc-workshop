"""LLM configuration endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.dependencies import reconfigure_llm
from src.api.schemas import LLMConfigureRequest, LLMValidateRequest
from src.models.llm import LLMConfiguration, LLMParameters, LLMProvider
from src.engines.llm_engine import LLMEngine

router = APIRouter(prefix="/llm", tags=["llm"])


@router.post("/configure")
def configure_llm(request: LLMConfigureRequest):
    """Configure the LLM provider."""
    config = LLMConfiguration(
        config_id="user_configured",
        provider=request.provider,
        model_name=request.model_name,
        api_key=request.api_key,
        api_endpoint=request.api_endpoint,
        parameters=LLMParameters(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ),
    )
    reconfigure_llm(config)
    return {"status": "configured", "provider": request.provider.value, "model": request.model_name}


@router.get("/providers")
def list_providers():
    """List available LLM providers."""
    return {
        "providers": [
            {"name": "openai", "description": "OpenAI GPT models"},
            {"name": "anthropic", "description": "Anthropic Claude models"},
            {"name": "local", "description": "Local models via Ollama or vLLM"},
        ]
    }


@router.post("/validate")
def validate_credentials(request: LLMValidateRequest):
    """Validate LLM provider credentials."""
    config = LLMConfiguration(
        config_id="validation_test",
        provider=request.provider,
        model_name=request.model_name,
        api_key=request.api_key,
        api_endpoint=request.api_endpoint,
    )
    try:
        engine = LLMEngine(configuration=config)
        is_valid = engine.adapter.validate_credentials()
        return {"valid": is_valid, "provider": request.provider.value}
    except Exception as e:
        return {"valid": False, "provider": request.provider.value, "error": str(e)}
