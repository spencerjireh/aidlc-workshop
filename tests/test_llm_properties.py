"""Property-based tests for LLM Engine."""

import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock, patch
import time

from src.models.llm import LLMConfiguration, LLMParameters, LLMProvider
from src.engines.llm_engine import LLMEngine


# Feature: llm-customer-segmentation-ads, Property 27: LLM Retry Logic
# **Validates: Requirements 9.4**
@given(prompt=st.text(min_size=10, max_size=100))
@settings(max_examples=100)
def test_llm_retry_logic(prompt):
    """Property: Failed LLM calls retry exactly 3 times with exponential backoff.
    
    For any LLM API call that fails, the system must retry exactly 3 times 
    with exponentially increasing delays before returning an error, and each 
    retry attempt must be logged.
    """
    # Create a mock adapter that always fails
    mock_adapter = Mock()
    mock_adapter.generate.side_effect = Exception("API Error")
    mock_adapter.get_provider_name.return_value = "MockProvider"
    
    # Create configuration
    config = LLMConfiguration(
        config_id="test_config",
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        api_key="test_key",
        parameters=LLMParameters()
    )
    
    # Create engine and replace adapter with mock
    engine = LLMEngine(config)
    engine.adapter = mock_adapter
    
    # Track sleep calls to verify exponential backoff
    sleep_times = []
    original_sleep = time.sleep
    
    def mock_sleep(seconds):
        sleep_times.append(seconds)
    
    with patch('time.sleep', side_effect=mock_sleep):
        # Call should fail after 3 retries
        with pytest.raises(Exception):
            engine.call_llm(prompt, max_retries=3)
    
    # Verify exactly 3 attempts were made
    assert mock_adapter.generate.call_count == 3
    
    # Verify exponential backoff: 2^0=1s, 2^1=2s (only 2 sleeps for 3 attempts)
    assert len(sleep_times) == 2
    assert sleep_times[0] == 1  # 2^0
    assert sleep_times[1] == 2  # 2^1
    
    # Verify all attempts were logged
    assert len(engine.request_log) == 3
    for i, log_entry in enumerate(engine.request_log):
        assert log_entry['attempt'] == i + 1
        assert log_entry['success'] is False
        assert 'error' in log_entry


# Feature: llm-customer-segmentation-ads, Property 26: LLM Parameter Configuration
# **Validates: Requirements 9.3**
@given(
    temperature=st.floats(min_value=0.0, max_value=2.0),
    max_tokens=st.integers(min_value=1, max_value=4000),
    top_p=st.floats(min_value=0.0, max_value=1.0),
    frequency_penalty=st.floats(min_value=-2.0, max_value=2.0),
    presence_penalty=st.floats(min_value=-2.0, max_value=2.0)
)
@settings(max_examples=100)
def test_llm_parameter_configuration(
    temperature, max_tokens, top_p, frequency_penalty, presence_penalty
):
    """Property: All LLM parameters are settable and retrievable with same values.
    
    For any LLM configuration, all parameters (temperature, max_tokens, top_p, 
    frequency_penalty, presence_penalty) must be settable and retrievable with 
    the same values.
    """
    # Create parameters with given values
    params = LLMParameters(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
    # Create configuration with these parameters
    config = LLMConfiguration(
        config_id="test_config",
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        api_key="test_key",
        parameters=params
    )
    
    # Verify all parameters are retrievable with same values
    assert config.parameters.temperature == temperature
    assert config.parameters.max_tokens == max_tokens
    assert config.parameters.top_p == top_p
    assert config.parameters.frequency_penalty == frequency_penalty
    assert config.parameters.presence_penalty == presence_penalty
    
    # Verify parameters can be serialized and deserialized
    params_dict = params.model_dump()
    restored_params = LLMParameters(**params_dict)
    
    assert restored_params.temperature == temperature
    assert restored_params.max_tokens == max_tokens
    assert restored_params.top_p == top_p
    assert restored_params.frequency_penalty == frequency_penalty
    assert restored_params.presence_penalty == presence_penalty
