"""Local model LLM provider adapter (Ollama/vLLM)."""

import logging
import requests
from typing import Optional

from .base import LLMProviderAdapter
from src.models.llm import LLMParameters

logger = logging.getLogger(__name__)


class LocalModelAdapter(LLMProviderAdapter):
    """Adapter for local LLM models (Ollama, vLLM, etc.)."""
    
    def __init__(self, endpoint: str, model_name: str = "llama2"):
        """Initialize local model adapter.
        
        Args:
            endpoint: API endpoint URL (e.g., "http://localhost:11434").
            model_name: Model name (e.g., "llama2", "mistral").
        """
        self.endpoint = endpoint.rstrip('/')
        self.model_name = model_name
    
    def validate_credentials(self) -> bool:
        """Validate local model endpoint and connection.
        
        Returns:
            bool: True if endpoint is accessible, False otherwise.
        """
        try:
            # Try to connect to the endpoint
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Local model endpoint validated: {self.endpoint}")
                return True
            else:
                logger.error(f"Local model endpoint returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Local model endpoint validation failed: {e}")
            return False
    
    def generate(self, prompt: str, parameters: LLMParameters) -> str:
        """Generate text using local model API.
        
        This implementation assumes Ollama-compatible API format.
        Adjust if using a different local model server.
        
        Args:
            prompt: Input prompt for generation.
            parameters: LLM generation parameters.
            
        Returns:
            str: Generated text response.
            
        Raises:
            requests.exceptions.RequestException: If API call fails.
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": parameters.temperature,
                    "num_predict": parameters.max_tokens,
                    "top_p": parameters.top_p,
                }
            }
            
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=120  # Local models can be slower
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.info(f"Local model generation successful")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Local model generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during local model generation: {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Return provider name.
        
        Returns:
            str: "Local"
        """
        return "Local"
