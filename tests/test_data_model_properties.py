"""Property-based tests for data model round-trip (Task 2.2, Property 1)."""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta

from src.models.customer import CustomerProfile, TransactionData
from src.models.segment import Segment, CustomerSegmentAssignment, ContributingFactor
from src.models.campaign import AdContent, AdFormat, Campaign, CampaignStatus
from src.models.chatbot import ChatbotResponse, ConversationContext, ChatMessage, MessageRole
from src.models.llm import LLMConfiguration, LLMParameters, LLMProvider


# Feature: llm-customer-segmentation-ads, Property 1: Data Ingestion Round-Trip
# Validates: Requirements 1.1, 1.2


@st.composite
def customer_profile_strategy(draw):
    """Generate a valid CustomerProfile."""
    return CustomerProfile(
        customer_id=draw(st.text(min_size=3, max_size=20, alphabet="abcdefghijklmnop0123456789")),
        age=draw(st.integers(min_value=18, max_value=80)),
        location=draw(st.text(min_size=2, max_size=30, alphabet="abcdefghijklmnop ")),
        transaction_frequency=draw(st.integers(min_value=1, max_value=200)),
        average_transaction_value=draw(st.floats(min_value=1.0, max_value=50000.0, allow_nan=False, allow_infinity=False)),
        merchant_categories=draw(st.lists(st.text(min_size=2, max_size=15, alphabet="abcdefghijklmnop"), min_size=1, max_size=5)),
        total_spend=draw(st.floats(min_value=1.0, max_value=500000.0, allow_nan=False, allow_infinity=False)),
        account_age_days=draw(st.integers(min_value=1, max_value=3650)),
        preferred_payment_methods=draw(st.lists(st.text(min_size=2, max_size=15, alphabet="abcdefghijklmnop"), min_size=1, max_size=3)),
        last_transaction_date=datetime.utcnow() - timedelta(days=draw(st.integers(min_value=0, max_value=365))),
    )


@given(customer=customer_profile_strategy())
@settings(max_examples=100)
def test_customer_profile_round_trip(customer):
    """Property: Serializing and deserializing a CustomerProfile preserves all fields."""
    data = customer.model_dump()
    restored = CustomerProfile(**data)

    assert restored.customer_id == customer.customer_id
    assert restored.age == customer.age
    assert restored.location == customer.location
    assert restored.transaction_frequency == customer.transaction_frequency
    assert abs(restored.average_transaction_value - customer.average_transaction_value) < 0.01
    assert restored.merchant_categories == customer.merchant_categories
    assert abs(restored.total_spend - customer.total_spend) < 0.01
    assert restored.account_age_days == customer.account_age_days
    assert restored.preferred_payment_methods == customer.preferred_payment_methods


@given(customer=customer_profile_strategy())
@settings(max_examples=100)
def test_customer_profile_json_round_trip(customer):
    """Property: JSON serialization and deserialization preserves all fields."""
    json_str = customer.model_dump_json()
    restored = CustomerProfile.model_validate_json(json_str)

    assert restored.customer_id == customer.customer_id
    assert restored.age == customer.age
    assert restored.transaction_frequency == customer.transaction_frequency
    assert restored.merchant_categories == customer.merchant_categories


@st.composite
def segment_strategy(draw):
    """Generate a valid Segment."""
    return Segment(
        segment_id=draw(st.text(min_size=3, max_size=15, alphabet="abcdef0123456789")),
        name=draw(st.text(min_size=3, max_size=30, alphabet="abcdefghijklmnop ")),
        description=draw(st.text(min_size=10, max_size=100, alphabet="abcdefghijklmnop ")),
        characteristics={"key": "value"},
        cluster_id=draw(st.integers(min_value=0, max_value=9)),
        centroid=draw(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=5)),
        size=draw(st.integers(min_value=1, max_value=10000)),
        average_transaction_value=draw(st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False)),
        transaction_frequency=draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False)),
        top_merchant_categories=draw(st.lists(st.text(min_size=2, max_size=15, alphabet="abcdefghijklmnop"), min_size=1, max_size=5)),
        differentiating_factors=draw(st.lists(st.text(min_size=3, max_size=30, alphabet="abcdefghijklmnop "), min_size=1, max_size=3)),
        pca_component_contributions={0: 0.5, 1: 0.3},
        version=draw(st.integers(min_value=1, max_value=10)),
    )


@given(segment=segment_strategy())
@settings(max_examples=100)
def test_segment_round_trip(segment):
    """Property: Serializing and deserializing a Segment preserves all fields."""
    data = segment.model_dump()
    restored = Segment(**data)

    assert restored.segment_id == segment.segment_id
    assert restored.name == segment.name
    assert restored.cluster_id == segment.cluster_id
    assert restored.size == segment.size
    assert restored.centroid == segment.centroid
    assert restored.version == segment.version


@given(
    temperature=st.floats(min_value=0.0, max_value=2.0),
    max_tokens=st.integers(min_value=1, max_value=4096),
)
@settings(max_examples=100)
def test_llm_parameters_round_trip(temperature, max_tokens):
    """Property: LLMParameters round-trip preserves all parameter values."""
    params = LLMParameters(temperature=temperature, max_tokens=max_tokens)
    data = params.model_dump()
    restored = LLMParameters(**data)

    assert abs(restored.temperature - temperature) < 1e-6
    assert restored.max_tokens == max_tokens
