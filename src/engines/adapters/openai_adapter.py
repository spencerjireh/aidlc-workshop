"""OpenAI LLM provider adapter."""

import logging
from typing import Optional
from openai import OpenAI
from openai import OpenAIError

from .base import LLMProviderAdapter
from src.models.llm import LLMParameters

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMProviderAdapter):
    """Adapter for OpenAI LLM provider."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        """Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key.
            model_name: Model name (e.g., "gpt-4", "gpt-3.5-turbo").
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client: Optional[OpenAI] = None
        
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.client = None
    
    def validate_credentials(self) -> bool:
        """Validate OpenAI API credentials and connection.
        
        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return False
        
        try:
            # Make a minimal API call to validate credentials
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            logger.info("OpenAI credentials validated successfully")
            return True
        except OpenAIError as e:
            logger.error(f"OpenAI credential validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI validation: {e}")
            return False
    
    def generate(self, prompt: str, parameters: LLMParameters) -> str:
        """Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt for generation.
            parameters: LLM generation parameters.
            
        Returns:
            str: Generated text response.
            
        Raises:
            ValueError: If client is not initialized.
            OpenAIError: If API call fails.
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=parameters.temperature,
                max_tokens=parameters.max_tokens,
                top_p=parameters.top_p,
                frequency_penalty=parameters.frequency_penalty,
                presence_penalty=parameters.presence_penalty
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"OpenAI generation successful. Tokens used: {response.usage.total_tokens}")
            
            return generated_text or ""
            
        except OpenAIError as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during OpenAI generation: {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Return provider name.
        
        Returns:
            str: "OpenAI"
        """
        return "OpenAI"
