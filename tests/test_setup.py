"""Test to verify the testing framework is properly configured."""

import pytest
from hypothesis import given, strategies as st


def test_pytest_works():
    """Verify pytest is working."""
    assert True


def test_imports():
    """Verify all key dependencies can be imported."""
    import fastapi
    import sklearn
    import pandas
    import numpy
    import openai
    import anthropic
    import hypothesis
    
    assert fastapi.__version__
    assert sklearn.__version__
    assert pandas.__version__
    assert numpy.__version__


@given(st.integers())
def test_hypothesis_works(x):
    """Verify hypothesis property-based testing is working."""
    assert isinstance(x, int)


def test_config_can_be_imported():
    """Verify configuration module can be imported."""
    from src.config import settings, LLMProvider
    
    assert settings.app_name == "LLM Customer Segmentation Ads"
    assert LLMProvider.OPENAI == "openai"
    assert LLMProvider.ANTHROPIC == "anthropic"
