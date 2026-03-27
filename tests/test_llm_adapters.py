"""Unit tests for LLM provider adapters."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.models.llm import LLMParameters
from src.engines.adapters import OpenAIAdapter, AnthropicAdapter, LocalModelAdapter


class TestOpenAIAdapter:
    """Test OpenAI adapter implementation."""
    
    def test_initialization(self):
        """Test OpenAI adapter initialization."""
        adapter = OpenAIAdapter(api_key="test_key", model_name="gpt-4")
        assert adapter.api_key == "test_key"
        assert adapter.model_name == "gpt-4"
        assert adapter.get_provider_name() == "OpenAI"
    
    def test_initialization_without_key(self):
        """Test OpenAI adapter initialization without API key."""
        adapter = OpenAIAdapter(api_key="", model_name="gpt-4")
        assert adapter.client is None
    
    @patch('src.engines.adapters.openai_adapter.OpenAI')
    def test_validate_credentials_success(self, mock_openai_class):
        """Test successful credential validation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        adapter = OpenAIAdapter(api_key="valid_key", model_name="gpt-4")
        assert adapter.validate_credentials() is True
    
    @patch('src.engines.adapters.openai_adapter.OpenAI')
    def test_validate_credentials_failure(self, mock_openai_class):
        """Test failed credential validation."""
        mock_client = Mock()
        from openai import OpenAIError
        mock_client.chat.completions.create.side_effect = OpenAIError("Invalid API key")
        mock_openai_class.return_value = mock_client
        
        adapter = OpenAIAdapter(api_key="invalid_key", model_name="gpt-4")
        assert adapter.validate_credentials() is False
    
    @patch('src.engines.adapters.openai_adapter.OpenAI')
    def test_generate_text(self, mock_openai_class):
        """Test text generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Generated text response"
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.usage = Mock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        adapter = OpenAIAdapter(api_key="test_key", model_name="gpt-4")
        params = LLMParameters(temperature=0.7, max_tokens=100)
        
        result = adapter.generate("Test prompt", params)
        
        assert result == "Generated text response"
        mock_client.chat.completions.create.assert_called_once()
    
    def test_generate_without_client(self):
        """Test generation fails without initialized client."""
        adapter = OpenAIAdapter(api_key="", model_name="gpt-4")
        params = LLMParameters()
        
        with pytest.raises(ValueError, match="OpenAI client not initialized"):
            adapter.generate("Test prompt", params)


class TestAnthropicAdapter:
    """Test Anthropic adapter implementation."""
    
    def test_initialization(self):
        """Test Anthropic adapter initialization."""
        adapter = AnthropicAdapter(api_key="test_key", model_name="claude-3-opus-20240229")
        assert adapter.api_key == "test_key"
        assert adapter.model_name == "claude-3-opus-20240229"
        assert adapter.get_provider_name() == "Anthropic"
    
    def test_initialization_without_key(self):
        """Test Anthropic adapter initialization without API key."""
        adapter = AnthropicAdapter(api_key="", model_name="claude-3-opus-20240229")
        assert adapter.client is None
    
    @patch('src.engines.adapters.anthropic_adapter.Anthropic')
    def test_validate_credentials_success(self, mock_anthropic_class):
        """Test successful credential validation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        adapter = AnthropicAdapter(api_key="valid_key", model_name="claude-3-opus-20240229")
        assert adapter.validate_credentials() is True
    
    @patch('src.engines.adapters.anthropic_adapter.Anthropic')
    def test_validate_credentials_failure(self, mock_anthropic_class):
        """Test failed credential validation."""
        mock_client = Mock()
        # APIError requires a message and request, so we'll use a generic Exception
        mock_client.messages.create.side_effect = Exception("Invalid API key")
        mock_anthropic_class.return_value = mock_client
        
        adapter = AnthropicAdapter(api_key="invalid_key", model_name="claude-3-opus-20240229")
        assert adapter.validate_credentials() is False
    
    @patch('src.engines.adapters.anthropic_adapter.Anthropic')
    def test_generate_text(self, mock_anthropic_class):
        """Test text generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_text_block = Mock()
        mock_text_block.text = "Generated text response"
        mock_response.content = [mock_text_block]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        adapter = AnthropicAdapter(api_key="test_key", model_name="claude-3-opus-20240229")
        params = LLMParameters(temperature=0.7, max_tokens=100)
        
        result = adapter.generate("Test prompt", params)
        
        assert result == "Generated text response"
        mock_client.messages.create.assert_called_once()
    
    def test_generate_without_client(self):
        """Test generation fails without initialized client."""
        adapter = AnthropicAdapter(api_key="", model_name="claude-3-opus-20240229")
        params = LLMParameters()
        
        with pytest.raises(ValueError, match="Anthropic client not initialized"):
            adapter.generate("Test prompt", params)


class TestLocalModelAdapter:
    """Test local model adapter implementation."""
    
    def test_initialization(self):
        """Test local model adapter initialization."""
        adapter = LocalModelAdapter(endpoint="http://localhost:11434", model_name="llama2")
        assert adapter.endpoint == "http://localhost:11434"
        assert adapter.model_name == "llama2"
        assert adapter.get_provider_name() == "Local"
    
    def test_endpoint_trailing_slash_removed(self):
        """Test that trailing slash is removed from endpoint."""
        adapter = LocalModelAdapter(endpoint="http://localhost:11434/", model_name="llama2")
        assert adapter.endpoint == "http://localhost:11434"
    
    @patch('src.engines.adapters.local_adapter.requests.get')
    def test_validate_credentials_success(self, mock_get):
        """Test successful endpoint validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        adapter = LocalModelAdapter(endpoint="http://localhost:11434", model_name="llama2")
        assert adapter.validate_credentials() is True
    
    @patch('src.engines.adapters.local_adapter.requests.get')
    def test_validate_credentials_failure(self, mock_get):
        """Test failed endpoint validation."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        adapter = LocalModelAdapter(endpoint="http://localhost:11434", model_name="llama2")
        assert adapter.validate_credentials() is False
    
    @patch('src.engines.adapters.local_adapter.requests.post')
    def test_generate_text(self, mock_post):
        """Test text generation."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "Generated text response"}
        mock_post.return_value = mock_response
        
        adapter = LocalModelAdapter(endpoint="http://localhost:11434", model_name="llama2")
        params = LLMParameters(temperature=0.7, max_tokens=100)
        
        result = adapter.generate("Test prompt", params)
        
        assert result == "Generated text response"
        mock_post.assert_called_once()
        
        # Verify request payload
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        assert payload['model'] == "llama2"
        assert payload['prompt'] == "Test prompt"
        assert payload['options']['temperature'] == 0.7
    
    @patch('src.engines.adapters.local_adapter.requests.post')
    def test_generate_handles_api_error(self, mock_post):
        """Test generation handles API errors."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        adapter = LocalModelAdapter(endpoint="http://localhost:11434", model_name="llama2")
        params = LLMParameters()
        
        with pytest.raises(requests.exceptions.RequestException):
            adapter.generate("Test prompt", params)
