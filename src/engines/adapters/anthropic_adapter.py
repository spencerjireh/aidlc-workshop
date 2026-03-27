"""Anthropic LLM provider adapter."""

import logging
from typing import Optional
from anthropic import Anthropic
from anthropic import APIError

from .base import LLMProviderAdapter
from src.models.llm import LLMParameters

logger = logging.getLogger(__name__)


class AnthropicAdapter(LLMProviderAdapter):
    """Adapter for Anthropic LLM provider."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        """Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key.
            model_name: Model name (e.g., "claude-3-opus-20240229", "claude-3-sonnet-20240229").
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client: Optional[Anthropic] = None
        
        if api_key:
            try:
                self.client = Anthropic(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
    
    def validate_credentials(self) -> bool:
        """Validate Anthropic API credentials and connection.
        
        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        if not self.client:
            logger.error("Anthropic client not initialized")
            return False
        
        try:
            # Make a minimal API call to validate credentials
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "test"}]
            )
            logger.info("Anthropic credentials validated successfully")
            return True
        except APIError as e:
            logger.error(f"Anthropic credential validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic validation: {e}")
            return False
    
    def generate(self, prompt: str, parameters: LLMParameters) -> str:
        """Generate text using Anthropic API.
        
        Args:
            prompt: Input prompt for generation.
            parameters: LLM generation parameters.
            
        Returns:
            str: Generated text response.
            
        Raises:
            ValueError: If client is not initialized.
            APIError: If API call fails.
        """
        if not self.client:
            raise ValueError("Anthropic client not initialized. Check API key.")
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=parameters.max_tokens,
                temperature=parameters.temperature,
                top_p=parameters.top_p,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text from response
            generated_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    generated_text += block.text
            
            logger.info(f"Anthropic generation successful. Tokens used: input={response.usage.input_tokens}, output={response.usage.output_tokens}")
            
            return generated_text
            
        except APIError as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Anthropic generation: {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Return provider name.
        
        Returns:
            str: "Anthropic"
        """
        return "Anthropic"
