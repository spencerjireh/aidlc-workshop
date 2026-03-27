"""Unit tests for LLM Engine."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.models.llm import LLMConfiguration, LLMParameters, LLMProvider
from src.models.ml import ClusterStatistics
from src.models.segment import SegmentProfile
from src.models.campaign import AdFormat
from src.models.chatbot import QueryIntent, QueryType, ConversationContext, ChatMessage, MessageRole
from src.models.customer import CustomerProfile
from src.engines.llm_engine import LLMEngine
from src.engines.adapters import OpenAIAdapter, AnthropicAdapter, LocalModelAdapter


class TestLLMEngineProviderSwitching:
    """Test provider adapter switching."""
    
    def test_create_openai_adapter(self):
        """Test OpenAI adapter creation."""
        config = LLMConfiguration(
            config_id="test_openai",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        assert isinstance(engine.adapter, OpenAIAdapter)
        assert engine.adapter.model_name == "gpt-4"
    
    def test_create_anthropic_adapter(self):
        """Test Anthropic adapter creation."""
        config = LLMConfiguration(
            config_id="test_anthropic",
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-opus-20240229",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        assert isinstance(engine.adapter, AnthropicAdapter)
        assert engine.adapter.model_name == "claude-3-opus-20240229"
    
    def test_create_local_adapter(self):
        """Test local model adapter creation."""
        config = LLMConfiguration(
            config_id="test_local",
            provider=LLMProvider.LOCAL,
            model_name="llama2",
            api_key="",
            api_endpoint="http://localhost:11434"
        )
        
        engine = LLMEngine(config)
        assert isinstance(engine.adapter, LocalModelAdapter)
        assert engine.adapter.model_name == "llama2"
    
    def test_local_adapter_requires_endpoint(self):
        """Test that local adapter requires api_endpoint."""
        config = LLMConfiguration(
            config_id="test_local",
            provider=LLMProvider.LOCAL,
            model_name="llama2",
            api_key=""
        )
        
        with pytest.raises(ValueError, match="api_endpoint is required"):
            LLMEngine(config)


class TestLLMEngineRetryLogic:
    """Test retry logic with mock failures."""
    
    def test_successful_call_on_first_attempt(self):
        """Test successful LLM call on first attempt."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = "Generated response"
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        result = engine.call_llm("Test prompt")
        
        assert result == "Generated response"
        assert mock_adapter.generate.call_count == 1
        assert len(engine.request_log) == 1
        assert engine.request_log[0]['success'] is True
    
    def test_retry_on_failure_then_success(self):
        """Test retry logic when first attempt fails but second succeeds."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        # First call fails, second succeeds
        mock_adapter.generate.side_effect = [
            Exception("API Error"),
            "Generated response"
        ]
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        with patch('time.sleep'):  # Skip actual sleep
            result = engine.call_llm("Test prompt", max_retries=3)
        
        assert result == "Generated response"
        assert mock_adapter.generate.call_count == 2
        assert len(engine.request_log) == 2
        assert engine.request_log[0]['success'] is False
        assert engine.request_log[1]['success'] is True
    
    def test_all_retries_fail(self):
        """Test that exception is raised after all retries fail."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.side_effect = Exception("API Error")
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        with patch('time.sleep'):  # Skip actual sleep
            with pytest.raises(Exception, match="API Error"):
                engine.call_llm("Test prompt", max_retries=3)
        
        assert mock_adapter.generate.call_count == 3
        assert len(engine.request_log) == 3
        assert all(log['success'] is False for log in engine.request_log)


class TestLLMEnginePromptConstruction:
    """Test prompt construction for different use cases."""
    
    def test_generate_segment_profile_prompt(self):
        """Test segment profile generation with proper prompt construction."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = """NAME: Young Urban Professionals
DESCRIPTION: Tech-savvy millennials with high transaction frequency and preference for digital payments.
DIFFERENTIATING_FACTORS: High income, Urban location, Frequent online shopping"""
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        cluster_stats = ClusterStatistics(
            cluster_id=0,
            size=250,
            average_age=28.5,
            location_distribution={"Manila": 150, "Quezon City": 100},
            average_transaction_frequency=15.2,
            average_transaction_value=1250.50,
            total_spend_distribution={"p50": 5000.0, "p75": 10000.0, "p95": 25000.0},
            top_merchant_categories=[("Food", 120), ("Transport", 80), ("Shopping", 50)],
            preferred_payment_methods={"Credit Card": 180, "Debit Card": 70}
        )
        
        profile = engine.generate_segment_profile(cluster_stats, 0)
        
        assert profile.name == "Young Urban Professionals"
        assert "Tech-savvy" in profile.description
        assert len(profile.differentiating_factors) == 3
        assert mock_adapter.generate.call_count == 1
        
        # Verify prompt includes cluster statistics
        call_args = mock_adapter.generate.call_args
        prompt = call_args[0][0]
        assert "Cluster ID: 0" in prompt
        assert "Size: 250" in prompt
        assert "Average Age: 28.5" in prompt
    
    def test_explain_cluster_assignment_prompt(self):
        """Test cluster assignment explanation with customer data."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = "This customer was assigned based on high transaction frequency and urban location."
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        customer = CustomerProfile(
            customer_id="cust_123",
            age=30,
            location="Manila",
            transaction_frequency=20,
            average_transaction_value=1500.0,
            merchant_categories=["Food", "Transport", "Shopping"],
            total_spend=30000.0,
            account_age_days=365,
            preferred_payment_methods=["Credit Card"],
            last_transaction_date=datetime.utcnow()
        )
        
        top_features = [
            ("transaction_frequency", 0.85),
            ("location", 0.72),
            ("average_transaction_value", 0.65)
        ]
        
        explanation = engine.explain_cluster_assignment(customer, [0.5, 0.3, 0.2], top_features)
        
        assert "transaction frequency" in explanation.lower()
        assert mock_adapter.generate.call_count == 1
        
        # Verify prompt includes customer data
        call_args = mock_adapter.generate.call_args
        prompt = call_args[0][0]
        assert "Age: 30" in prompt
        assert "Location: Manila" in prompt
    
    def test_generate_ad_content_prompt(self):
        """Test ad content generation with format constraints."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = """AD1: Get 10% cashback on all purchases!
AD2: Earn rewards with every transaction today!
AD3: Save more with exclusive cashback offers!"""
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        segment_profile = SegmentProfile(
            segment_id="seg_1",
            name="Budget Shoppers",
            description="Price-conscious customers who love deals",
            differentiating_factors=["Low income", "High deal sensitivity", "Frequent small purchases"]
        )
        
        ads = engine.generate_ad_content(
            segment_profile,
            "Promote cashback offers for budget-conscious shoppers",
            AdFormat.SHORT,
            num_variations=3
        )
        
        assert len(ads) == 3
        for ad in ads:
            assert len(ad.content) <= 50  # SHORT format limit
            assert ad.format == AdFormat.SHORT
            assert ad.segment_id == "seg_1"
        
        # Verify prompt includes segment profile
        call_args = mock_adapter.generate.call_args
        prompt = call_args[0][0]
        assert "Budget Shoppers" in prompt
        assert "50 characters" in prompt


class TestLLMEngineErrorHandling:
    """Test error handling for API failures."""
    
    def test_handle_openai_api_error(self):
        """Test handling of OpenAI API errors."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.side_effect = Exception("Rate limit exceeded")
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        with patch('time.sleep'):
            with pytest.raises(Exception, match="Rate limit exceeded"):
                engine.call_llm("Test prompt", max_retries=2)
        
        # Verify error was logged
        assert len(engine.request_log) == 2
        assert all('error' in log for log in engine.request_log)
    
    def test_handle_invalid_response_format(self):
        """Test handling of invalid LLM response format."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        # Return response without expected format
        mock_adapter.generate.return_value = "Invalid response format"
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        cluster_stats = ClusterStatistics(
            cluster_id=0,
            size=100,
            average_age=30.0,
            location_distribution={"Manila": 100},
            average_transaction_frequency=10.0,
            average_transaction_value=1000.0,
            total_spend_distribution={"p50": 5000.0},
            top_merchant_categories=[("Food", 50)],
            preferred_payment_methods={"Cash": 100}
        )
        
        # Should handle gracefully with default values
        profile = engine.generate_segment_profile(cluster_stats, 0)
        
        assert profile.name == "Unnamed Segment"
        assert profile.description == "No description available"


class TestLLMEngineChatbotIntegration:
    """Test chatbot query interpretation and response generation."""
    
    def test_interpret_query_intent(self):
        """Test query intent interpretation."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = """TYPE: customer_count
ENTITIES: segment_name=Young Professionals
CONFIDENCE: 0.9"""
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        context = ConversationContext(
            session_id="session_1",
            user_id="user_1",
            conversation_history=[]
        )
        
        intent = engine.interpret_query("How many customers are in Young Professionals segment?", context)
        
        assert intent.intent_type == QueryType.CUSTOMER_COUNT
        assert "segment_name" in intent.entities
        assert intent.confidence == 0.9
    
    def test_generate_chatbot_response(self):
        """Test chatbot response generation."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = """RESPONSE: The Young Professionals segment has 250 customers with an average age of 28.5 years.
FOLLOWUPS: What are the top merchant categories?, How does this compare to other segments?, Show me the transaction patterns
VISUALIZATION: table"""
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        intent = QueryIntent(
            intent_type=QueryType.CUSTOMER_COUNT,
            entities={"segment_name": "Young Professionals"},
            confidence=0.9
        )
        
        data = {
            "segment_name": "Young Professionals",
            "customer_count": 250,
            "average_age": 28.5
        }
        
        context = ConversationContext(
            session_id="session_1",
            user_id="user_1",
            conversation_history=[]
        )
        
        response = engine.generate_response(intent, data, context)
        
        assert "250 customers" in response.text
        assert len(response.suggested_followups) == 3
        assert response.visualization_type == "table"
        assert response.session_id == "session_1"


class TestLLMEngineRequestLogging:
    """Test LLM request/response logging."""
    
    def test_log_successful_request(self):
        """Test logging of successful LLM requests."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.return_value = "Response"
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        engine.call_llm("Test prompt")
        
        logs = engine.get_request_log()
        assert len(logs) == 1
        assert logs[0]['success'] is True
        assert logs[0]['provider'] == "OpenAI"
        assert logs[0]['model'] == "gpt-4"
        assert 'timestamp' in logs[0]
        assert 'elapsed_time_seconds' in logs[0]
    
    def test_log_failed_request(self):
        """Test logging of failed LLM requests."""
        config = LLMConfiguration(
            config_id="test",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="test_key"
        )
        
        engine = LLMEngine(config)
        mock_adapter = Mock()
        mock_adapter.generate.side_effect = Exception("API Error")
        mock_adapter.get_provider_name.return_value = "OpenAI"
        engine.adapter = mock_adapter
        
        with patch('time.sleep'):
            with pytest.raises(Exception):
                engine.call_llm("Test prompt", max_retries=2)
        
        logs = engine.get_request_log()
        assert len(logs) == 2
        assert all(log['success'] is False for log in logs)
        assert all('error' in log for log in logs)
