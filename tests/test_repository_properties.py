"""Property-based tests for data repositories."""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta
import hashlib

from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository
from src.models.customer import CustomerProfile, TransactionData
from src.models.segment import Segment, CustomerSegmentAssignment, ContributingFactor
from src.models.campaign import Campaign, AdContent, AdPerformanceMetrics, AdFormat, CampaignStatus


# Strategy for generating customer profiles
@st.composite
def customer_profile_strategy(draw):
    """Generate a valid CustomerProfile."""
    customer_id = draw(st.text(min_size=5, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    age = draw(st.integers(min_value=18, max_value=80))
    location = draw(st.text(min_size=3, max_size=20))
    transaction_frequency = draw(st.integers(min_value=1, max_value=100))
    average_transaction_value = draw(st.floats(min_value=10.0, max_value=10000.0))
    merchant_categories = draw(st.lists(st.text(min_size=3, max_size=15), min_size=1, max_size=5))
    total_spend = draw(st.floats(min_value=100.0, max_value=100000.0))
    account_age_days = draw(st.integers(min_value=1, max_value=3650))
    preferred_payment_methods = draw(st.lists(st.text(min_size=3, max_size=15), min_size=1, max_size=3))
    last_transaction_date = datetime.utcnow() - timedelta(days=draw(st.integers(min_value=0, max_value=365)))
    
    return CustomerProfile(
        customer_id=customer_id,
        age=age,
        location=location,
        transaction_frequency=transaction_frequency,
        average_transaction_value=average_transaction_value,
        merchant_categories=merchant_categories,
        total_spend=total_spend,
        account_age_days=account_age_days,
        preferred_payment_methods=preferred_payment_methods,
        last_transaction_date=last_transaction_date,
    )


# Feature: llm-customer-segmentation-ads, Property 22: PII Anonymization
@given(customer=customer_profile_strategy())
@settings(max_examples=100)
def test_pii_anonymization(customer):
    """
    **Validates: Requirements 8.1**
    
    Property: For any customer profile processed by the system, after anonymization,
    the customer_id must be a non-reversible hash.
    """
    repo = CustomerDataRepository()
    
    # Store customer (which anonymizes the ID)
    anonymized_id = repo.create_customer(customer)
    
    # Verify the ID is anonymized (hashed)
    assert anonymized_id != customer.customer_id
    
    # Verify it's a hash (16 character hex string from SHA-256)
    assert len(anonymized_id) == 16
    assert all(c in '0123456789abcdef' for c in anonymized_id)
    
    # Verify it's deterministic (same input produces same hash)
    expected_hash = hashlib.sha256(customer.customer_id.encode()).hexdigest()[:16]
    assert anonymized_id == expected_hash
    
    # Verify we can retrieve the customer with the anonymized ID
    retrieved = repo.get_customer(anonymized_id)
    assert retrieved is not None
    assert retrieved.customer_id == anonymized_id


# Feature: llm-customer-segmentation-ads, Property 23: Encryption Round-Trip
@given(customer=customer_profile_strategy())
@settings(max_examples=100)
def test_encryption_round_trip(customer):
    """
    **Validates: Requirements 8.2**
    
    Property: For any customer data, encrypting and then decrypting the data
    should produce data equivalent to the original.
    """
    repo = CustomerDataRepository()
    
    # Store customer (encrypts data)
    anonymized_id = repo.create_customer(customer)
    
    # Retrieve customer (decrypts data)
    retrieved = repo.get_customer(anonymized_id)
    
    # Verify all fields match (except customer_id which is anonymized)
    assert retrieved is not None
    assert retrieved.age == customer.age
    assert retrieved.location == customer.location
    assert retrieved.transaction_frequency == customer.transaction_frequency
    assert abs(retrieved.average_transaction_value - customer.average_transaction_value) < 0.01
    assert retrieved.merchant_categories == customer.merchant_categories
    assert abs(retrieved.total_spend - customer.total_spend) < 0.01
    assert retrieved.account_age_days == customer.account_age_days
    assert retrieved.preferred_payment_methods == customer.preferred_payment_methods
    # Note: datetime comparison may have microsecond differences due to serialization
    assert abs((retrieved.last_transaction_date - customer.last_transaction_date).total_seconds()) < 1


# Additional property test: Multiple customers encryption independence
@given(customers=st.lists(customer_profile_strategy(), min_size=2, max_size=10, unique_by=lambda c: c.customer_id))
@settings(max_examples=50)
def test_multiple_customers_encryption_independence(customers):
    """
    Property: Encrypting multiple customers should produce independent encrypted data.
    Each customer should be retrievable with correct data.
    """
    repo = CustomerDataRepository()
    
    # Store all customers
    anonymized_ids = []
    for customer in customers:
        anonymized_id = repo.create_customer(customer)
        anonymized_ids.append(anonymized_id)
    
    # Verify all customers are retrievable with correct data
    for i, customer in enumerate(customers):
        retrieved = repo.get_customer(anonymized_ids[i])
        assert retrieved is not None
        assert retrieved.age == customer.age
        assert retrieved.location == customer.location


# Property test: Segment versioning
@given(
    segment_id=st.text(min_size=5, max_size=20),
    versions=st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_segment_versioning_tracking(segment_id, versions):
    """
    **Validates: Requirements 10.5**
    
    Property: For any segment that is updated multiple times, version history
    should track all versions with incrementing version numbers.
    """
    repo = SegmentDataRepository()
    
    # Create initial segment
    segment = Segment(
        segment_id=segment_id,
        name="Test Segment",
        description="Initial description",
        characteristics={"key": "value"},
        cluster_id=0,
        centroid=[1.0, 2.0, 3.0],
        size=100,
        average_transaction_value=500.0,
        transaction_frequency=10.0,
        top_merchant_categories=["Food", "Transport"],
        differentiating_factors=["High frequency"],
        pca_component_contributions={0: 0.5, 1: 0.3},
        version=1
    )
    
    repo.create_segment(segment)
    
    # Update segment multiple times
    for v in range(2, versions + 1):
        segment.name = f"Test Segment v{v}"
        segment.description = f"Description v{v}"
        repo.update_segment(segment_id, segment)
    
    # Verify version history
    history = repo.get_segment_version_history(segment_id)
    assert len(history) >= versions
    
    # Verify version numbers are tracked
    version_numbers = [h['version'] for h in history]
    assert 1 in version_numbers  # Initial version
    
    # Verify current segment has latest version
    current = repo.get_segment(segment_id)
    assert current is not None
    assert current.version == versions


# Property test: Campaign-segment associations
@given(
    campaign_id=st.text(min_size=5, max_size=20),
    segment_ids=st.lists(st.text(min_size=5, max_size=20), min_size=1, max_size=5, unique=True)
)
@settings(max_examples=50)
def test_campaign_segment_associations(campaign_id, segment_ids):
    """
    **Validates: Requirements 5.1, 5.3**
    
    Property: For any campaign with target segments, associations must exist
    between the campaign and all selected segments.
    """
    repo = CampaignDataRepository()
    
    # Create campaign
    campaign = Campaign(
        campaign_id=campaign_id,
        name="Test Campaign",
        target_segment_ids=segment_ids,
        ad_content_ids=["ad1", "ad2"],
        estimated_reach=1000,
    )
    
    repo.create_campaign(campaign)
    
    # Verify campaign can be retrieved
    retrieved = repo.get_campaign(campaign_id)
    assert retrieved is not None
    assert set(retrieved.target_segment_ids) == set(segment_ids)
    
    # Verify associations exist for all segments
    for segment_id in segment_ids:
        campaigns = repo.get_campaigns_by_segment(segment_id)
        assert any(c.campaign_id == campaign_id for c in campaigns)


# Property test: Performance metrics storage
@given(
    ad_id=st.text(min_size=5, max_size=20),
    impressions=st.integers(min_value=100, max_value=10000),
    clicks=st.integers(min_value=0, max_value=100)
)
@settings(max_examples=50)
def test_performance_metrics_storage(ad_id, impressions, clicks):
    """
    **Validates: Requirements 7.2**
    
    Property: For any ad performance metrics, storing and retrieving should
    preserve all metric values.
    """
    # Ensure clicks don't exceed impressions
    clicks = min(clicks, impressions)
    conversions = min(clicks // 2, clicks)
    
    repo = CampaignDataRepository()
    
    # Create metrics
    metrics = AdPerformanceMetrics(
        ad_id=ad_id,
        impressions=impressions,
        clicks=clicks,
        conversions=conversions,
        click_through_rate=clicks / impressions if impressions > 0 else 0.0,
        conversion_rate=conversions / clicks if clicks > 0 else 0.0,
        segment_id="seg1",
        measurement_period_start=datetime.utcnow() - timedelta(days=7),
        measurement_period_end=datetime.utcnow(),
    )
    
    # Store metrics
    repo.store_performance_metrics(metrics)
    
    # Retrieve and verify
    retrieved = repo.get_performance_metrics(ad_id)
    assert retrieved is not None
    assert retrieved.ad_id == ad_id
    assert retrieved.impressions == impressions
    assert retrieved.clicks == clicks
    assert retrieved.conversions == conversions
    assert abs(retrieved.click_through_rate - metrics.click_through_rate) < 0.0001
