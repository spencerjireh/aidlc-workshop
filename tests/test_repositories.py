"""Unit tests for data repositories."""

import pytest
from datetime import datetime, timedelta
import os

from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository
from src.models.customer import CustomerProfile, TransactionData
from src.models.segment import Segment, CustomerSegmentAssignment, ContributingFactor
from src.models.campaign import Campaign, AdContent, AdPerformanceMetrics, AdFormat, CampaignStatus


class TestCustomerDataRepository:
    """Unit tests for CustomerDataRepository."""
    
    def test_create_and_retrieve_customer(self):
        """Test creating and retrieving a customer profile."""
        repo = CustomerDataRepository()
        
        customer = CustomerProfile(
            customer_id="cust_001",
            age=30,
            location="Manila",
            transaction_frequency=50,
            average_transaction_value=1500.0,
            merchant_categories=["Food", "Transport"],
            total_spend=75000.0,
            account_age_days=365,
            preferred_payment_methods=["Credit Card", "E-wallet"],
            last_transaction_date=datetime.utcnow(),
        )
        
        anonymized_id = repo.create_customer(customer)
        
        # Verify anonymized ID is different from original
        assert anonymized_id != customer.customer_id
        
        # Retrieve and verify
        retrieved = repo.get_customer(anonymized_id)
        assert retrieved is not None
        assert retrieved.age == 30
        assert retrieved.location == "Manila"
    
    def test_update_customer(self):
        """Test updating a customer profile."""
        repo = CustomerDataRepository()
        
        customer = CustomerProfile(
            customer_id="cust_002",
            age=25,
            location="Quezon City",
            transaction_frequency=30,
            average_transaction_value=800.0,
            merchant_categories=["Shopping"],
            total_spend=24000.0,
            account_age_days=180,
            preferred_payment_methods=["Debit Card"],
            last_transaction_date=datetime.utcnow(),
        )
        
        anonymized_id = repo.create_customer(customer)
        
        # Update customer
        customer.age = 26
        customer.location = "Makati"
        success = repo.update_customer(anonymized_id, customer)
        
        assert success is True
        
        # Verify update
        retrieved = repo.get_customer(anonymized_id)
        assert retrieved.age == 26
        assert retrieved.location == "Makati"
    
    def test_delete_customer(self):
        """Test deleting a customer profile."""
        repo = CustomerDataRepository()
        
        customer = CustomerProfile(
            customer_id="cust_003",
            age=35,
            location="Pasig",
            transaction_frequency=20,
            average_transaction_value=2000.0,
            merchant_categories=["Utilities"],
            total_spend=40000.0,
            account_age_days=730,
            preferred_payment_methods=["Bank Transfer"],
            last_transaction_date=datetime.utcnow(),
        )
        
        anonymized_id = repo.create_customer(customer)
        
        # Delete customer
        success = repo.delete_customer(anonymized_id)
        assert success is True
        
        # Verify deletion
        retrieved = repo.get_customer(anonymized_id)
        assert retrieved is None
    
    def test_list_customers_with_pagination(self):
        """Test listing customers with pagination."""
        repo = CustomerDataRepository()
        
        # Create multiple customers
        for i in range(5):
            customer = CustomerProfile(
                customer_id=f"cust_{i:03d}",
                age=20 + i,
                location=f"Location {i}",
                transaction_frequency=10 + i,
                average_transaction_value=500.0 + i * 100,
                merchant_categories=["Category"],
                total_spend=5000.0 + i * 1000,
                account_age_days=100 + i * 50,
                preferred_payment_methods=["Method"],
                last_transaction_date=datetime.utcnow(),
            )
            repo.create_customer(customer)
        
        # Test pagination
        customers = repo.list_customers(limit=3, offset=0)
        assert len(customers) == 3
        
        customers = repo.list_customers(limit=2, offset=3)
        assert len(customers) == 2
    
    def test_add_and_retrieve_transactions(self):
        """Test adding and retrieving customer transactions."""
        repo = CustomerDataRepository()
        
        customer = CustomerProfile(
            customer_id="cust_004",
            age=28,
            location="BGC",
            transaction_frequency=40,
            average_transaction_value=1200.0,
            merchant_categories=["Dining"],
            total_spend=48000.0,
            account_age_days=200,
            preferred_payment_methods=["E-wallet"],
            last_transaction_date=datetime.utcnow(),
        )
        
        anonymized_id = repo.create_customer(customer)
        
        # Add transactions
        transaction1 = TransactionData(
            transaction_id="txn_001",
            customer_id="cust_004",
            amount=150.0,
            merchant_category="Food",
            merchant_name="Restaurant A",
            transaction_type="payment",
            timestamp=datetime.utcnow(),
            payment_method="E-wallet",
            location="BGC",
        )
        
        transaction2 = TransactionData(
            transaction_id="txn_002",
            customer_id="cust_004",
            amount=200.0,
            merchant_category="Food",
            merchant_name="Restaurant B",
            transaction_type="payment",
            timestamp=datetime.utcnow(),
            payment_method="E-wallet",
            location="BGC",
        )
        
        repo.add_transaction(transaction1)
        repo.add_transaction(transaction2)
        
        # Retrieve transactions
        transactions = repo.get_customer_transactions("cust_004")
        assert len(transactions) == 2
        assert transactions[0].amount == 150.0
        assert transactions[1].amount == 200.0
    
    def test_customer_exists(self):
        """Test checking if a customer exists."""
        repo = CustomerDataRepository()
        
        customer = CustomerProfile(
            customer_id="cust_005",
            age=32,
            location="Taguig",
            transaction_frequency=25,
            average_transaction_value=1800.0,
            merchant_categories=["Entertainment"],
            total_spend=45000.0,
            account_age_days=300,
            preferred_payment_methods=["Credit Card"],
            last_transaction_date=datetime.utcnow(),
        )
        
        anonymized_id = repo.create_customer(customer)
        
        assert repo.customer_exists("cust_005") is True
        assert repo.customer_exists("nonexistent") is False
    


class TestSegmentDataRepository:
    """Unit tests for SegmentDataRepository."""
    
    def test_create_and_retrieve_segment(self):
        """Test creating and retrieving a segment."""
        repo = SegmentDataRepository()
        
        segment = Segment(
            segment_id="seg_001",
            name="High-Value Shoppers",
            description="Customers with high transaction values",
            characteristics={"avg_value": "high", "frequency": "medium"},
            cluster_id=0,
            centroid=[1.5, 2.3, 0.8],
            size=250,
            average_transaction_value=2500.0,
            transaction_frequency=45.0,
            top_merchant_categories=["Shopping", "Dining"],
            differentiating_factors=["High spending", "Urban location"],
            pca_component_contributions={0: 0.6, 1: 0.3, 2: 0.1},
            version=1,
        )
        
        segment_id = repo.create_segment(segment)
        
        # Retrieve and verify
        retrieved = repo.get_segment(segment_id)
        assert retrieved is not None
        assert retrieved.name == "High-Value Shoppers"
        assert retrieved.size == 250
    
    def test_update_segment_increments_version(self):
        """Test that updating a segment increments version."""
        repo = SegmentDataRepository()
        
        segment = Segment(
            segment_id="seg_002",
            name="Budget Shoppers",
            description="Cost-conscious customers",
            characteristics={"avg_value": "low", "frequency": "high"},
            cluster_id=1,
            centroid=[0.5, 1.2, 0.3],
            size=400,
            average_transaction_value=500.0,
            transaction_frequency=60.0,
            top_merchant_categories=["Groceries", "Transport"],
            differentiating_factors=["Low spending", "High frequency"],
            pca_component_contributions={0: 0.4, 1: 0.4, 2: 0.2},
            version=1,
        )
        
        repo.create_segment(segment)
        
        # Update segment
        segment.name = "Budget-Conscious Shoppers"
        repo.update_segment("seg_002", segment)
        
        # Verify version incremented
        retrieved = repo.get_segment("seg_002")
        assert retrieved.version == 2
        assert retrieved.name == "Budget-Conscious Shoppers"
    
    def test_assign_customer_to_segment(self):
        """Test assigning a customer to a segment."""
        repo = SegmentDataRepository()
        
        # Create segment
        segment = Segment(
            segment_id="seg_003",
            name="Test Segment",
            description="Test description",
            characteristics={},
            cluster_id=0,
            centroid=[1.0, 2.0],
            size=100,
            average_transaction_value=1000.0,
            transaction_frequency=30.0,
            top_merchant_categories=["Food"],
            differentiating_factors=["Factor"],
            pca_component_contributions={0: 0.5},
            version=1,
        )
        repo.create_segment(segment)
        
        # Create assignment
        assignment = CustomerSegmentAssignment(
            assignment_id="assign_001",
            customer_id="cust_001",
            segment_id="seg_003",
            confidence_score=0.85,
            distance_to_centroid=0.5,
        )
        
        assignment_id = repo.assign_customer_to_segment(assignment)
        
        # Verify assignment
        retrieved = repo.get_customer_assignment("cust_001")
        assert retrieved is not None
        assert retrieved.segment_id == "seg_003"
        assert retrieved.confidence_score == 0.85
    
    def test_get_segment_customers(self):
        """Test getting all customers in a segment."""
        repo = SegmentDataRepository()
        
        # Create segment
        segment = Segment(
            segment_id="seg_004",
            name="Test Segment",
            description="Test",
            characteristics={},
            cluster_id=0,
            centroid=[1.0],
            size=3,
            average_transaction_value=1000.0,
            transaction_frequency=30.0,
            top_merchant_categories=["Food"],
            differentiating_factors=["Factor"],
            pca_component_contributions={0: 0.5},
            version=1,
        )
        repo.create_segment(segment)
        
        # Assign multiple customers
        for i in range(3):
            assignment = CustomerSegmentAssignment(
                assignment_id=f"assign_{i}",
                customer_id=f"cust_{i}",
                segment_id="seg_004",
                confidence_score=0.8,
                distance_to_centroid=0.3,
            )
            repo.assign_customer_to_segment(assignment)
        
        # Get customers
        customers = repo.get_segment_customers("seg_004")
        assert len(customers) == 3
    
    def test_segment_version_history(self):
        """Test segment version history tracking."""
        repo = SegmentDataRepository()
        
        segment = Segment(
            segment_id="seg_005",
            name="Version Test",
            description="Initial",
            characteristics={},
            cluster_id=0,
            centroid=[1.0],
            size=100,
            average_transaction_value=1000.0,
            transaction_frequency=30.0,
            top_merchant_categories=["Food"],
            differentiating_factors=["Factor"],
            pca_component_contributions={0: 0.5},
            version=1,
        )
        
        repo.create_segment(segment)
        
        # Update multiple times
        for i in range(3):
            segment.description = f"Version {i + 2}"
            repo.update_segment("seg_005", segment)
        
        # Check version history
        history = repo.get_segment_version_history("seg_005")
        assert len(history) >= 4  # Initial + 3 updates
    
    def test_rollback_segment_to_version(self):
        """Test rolling back a segment to a previous version."""
        repo = SegmentDataRepository()
        
        segment = Segment(
            segment_id="seg_006",
            name="Rollback Test",
            description="Version 1",
            characteristics={},
            cluster_id=0,
            centroid=[1.0],
            size=100,
            average_transaction_value=1000.0,
            transaction_frequency=30.0,
            top_merchant_categories=["Food"],
            differentiating_factors=["Factor"],
            pca_component_contributions={0: 0.5},
            version=1,
        )
        
        repo.create_segment(segment)
        
        # Update to version 2
        segment.description = "Version 2"
        repo.update_segment("seg_006", segment)
        
        # Rollback to version 1
        success = repo.rollback_segment_to_version("seg_006", 1)
        assert success is True
        
        # Verify rollback
        retrieved = repo.get_segment("seg_006")
        assert retrieved.description == "Version 1"


class TestCampaignDataRepository:
    """Unit tests for CampaignDataRepository."""
    
    def test_create_and_retrieve_campaign(self):
        """Test creating and retrieving a campaign."""
        repo = CampaignDataRepository()
        
        campaign = Campaign(
            campaign_id="camp_001",
            name="Summer Sale",
            target_segment_ids=["seg_001", "seg_002"],
            ad_content_ids=["ad_001", "ad_002"],
            status=CampaignStatus.DRAFT,
            estimated_reach=5000,
        )
        
        campaign_id = repo.create_campaign(campaign)
        
        # Retrieve and verify
        retrieved = repo.get_campaign(campaign_id)
        assert retrieved is not None
        assert retrieved.name == "Summer Sale"
        assert len(retrieved.target_segment_ids) == 2
    
    def test_update_campaign_status(self):
        """Test updating campaign status."""
        repo = CampaignDataRepository()
        
        campaign = Campaign(
            campaign_id="camp_002",
            name="Flash Sale",
            target_segment_ids=["seg_001"],
            ad_content_ids=["ad_001"],
            status=CampaignStatus.DRAFT,
            estimated_reach=2000,
        )
        
        repo.create_campaign(campaign)
        
        # Update status
        campaign.status = CampaignStatus.ACTIVE
        repo.update_campaign("camp_002", campaign)
        
        # Verify update
        retrieved = repo.get_campaign("camp_002")
        assert retrieved.status == CampaignStatus.ACTIVE
    
    def test_get_campaigns_by_segment(self):
        """Test getting campaigns targeting a specific segment."""
        repo = CampaignDataRepository()
        
        # Create campaigns targeting same segment
        for i in range(3):
            campaign = Campaign(
                campaign_id=f"camp_{i}",
                name=f"Campaign {i}",
                target_segment_ids=["seg_shared"],
                ad_content_ids=[f"ad_{i}"],
                estimated_reach=1000,
            )
            repo.create_campaign(campaign)
        
        # Get campaigns for segment
        campaigns = repo.get_campaigns_by_segment("seg_shared")
        assert len(campaigns) == 3
    
    def test_create_and_retrieve_ad_content(self):
        """Test creating and retrieving ad content."""
        repo = CampaignDataRepository()
        
        ad = AdContent(
            ad_id="ad_001",
            segment_id="seg_001",
            campaign_id="camp_001",
            format=AdFormat.SHORT,
            content="Save big on your next purchase!",
            variation_number=1,
            use_case="cashback",
        )
        
        ad_id = repo.create_ad_content(ad)
        
        # Retrieve and verify
        retrieved = repo.get_ad_content(ad_id)
        assert retrieved is not None
        assert retrieved.content == "Save big on your next purchase!"
        assert retrieved.format == AdFormat.SHORT
    
    def test_store_and_retrieve_performance_metrics(self):
        """Test storing and retrieving performance metrics."""
        repo = CampaignDataRepository()
        
        metrics = AdPerformanceMetrics(
            ad_id="ad_001",
            impressions=10000,
            clicks=500,
            conversions=50,
            click_through_rate=0.05,
            conversion_rate=0.1,
            segment_id="seg_001",
            measurement_period_start=datetime.utcnow() - timedelta(days=7),
            measurement_period_end=datetime.utcnow(),
        )
        
        repo.store_performance_metrics(metrics)
        
        # Retrieve and verify
        retrieved = repo.get_performance_metrics("ad_001")
        assert retrieved is not None
        assert retrieved.impressions == 10000
        assert retrieved.clicks == 500
        assert retrieved.click_through_rate == 0.05
    
    def test_get_campaign_performance(self):
        """Test getting performance metrics for all ads in a campaign."""
        repo = CampaignDataRepository()
        
        # Create campaign with ads
        campaign = Campaign(
            campaign_id="camp_perf",
            name="Performance Test",
            target_segment_ids=["seg_001"],
            ad_content_ids=["ad_p1", "ad_p2"],
            estimated_reach=1000,
        )
        repo.create_campaign(campaign)
        
        # Add performance metrics for ads
        for i, ad_id in enumerate(["ad_p1", "ad_p2"]):
            metrics = AdPerformanceMetrics(
                ad_id=ad_id,
                impressions=1000 * (i + 1),
                clicks=50 * (i + 1),
                conversions=5 * (i + 1),
                click_through_rate=0.05,
                conversion_rate=0.1,
                segment_id="seg_001",
                measurement_period_start=datetime.utcnow() - timedelta(days=7),
                measurement_period_end=datetime.utcnow(),
            )
            repo.store_performance_metrics(metrics)
        
        # Get campaign performance
        performance = repo.get_campaign_performance("camp_perf")
        assert len(performance) == 2
    
    def test_delete_campaign(self):
        """Test deleting a campaign."""
        repo = CampaignDataRepository()
        
        campaign = Campaign(
            campaign_id="camp_delete",
            name="Delete Test",
            target_segment_ids=["seg_001"],
            ad_content_ids=["ad_001"],
            estimated_reach=500,
        )
        
        repo.create_campaign(campaign)
        
        # Delete campaign
        success = repo.delete_campaign("camp_delete")
        assert success is True
        
        # Verify deletion
        retrieved = repo.get_campaign("camp_delete")
        assert retrieved is None
