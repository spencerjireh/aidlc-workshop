"""Abstract base class for LLM provider adapters."""

from abc import ABC, abstractmethod
from src.models.llm import LLMParameters


class LLMProviderAdapter(ABC):
    """Abstract base class for LLM provider adapters.
    
    All LLM provider implementations must inherit from this class
    and implement the required abstract methods.
    """
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate API credentials and connection.
        
        Returns:
            bool: True if credentials are valid and connection successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, parameters: LLMParameters) -> str:
        """Generate text from prompt using the LLM provider.
        
        Args:
            prompt: The input prompt for text generation.
            parameters: LLM generation parameters (temperature, max_tokens, etc.).
            
        Returns:
            str: Generated text response from the LLM.
            
        Raises:
            Exception: If generation fails or API call encounters an error.
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name.
        
        Returns:
            str: Provider name (e.g., "OpenAI", "Anthropic", "Local").
        """
        pass
