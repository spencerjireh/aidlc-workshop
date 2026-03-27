"""Unit tests for data models."""

import pytest
from datetime import datetime
from src.models import (
    CustomerProfile,
    TransactionData,
    Segment,
    CustomerSegmentAssignment,
    ContributingFactor,
    PCAResult,
    ClusteringResult,
    ClusterStatistics,
    AdContent,
    AdFormat,
    Campaign,
    CampaignStatus,
    LLMConfiguration,
    LLMProvider,
    LLMParameters,
    ConversationContext,
    ChatMessage,
    MessageRole,
    QueryIntent,
    QueryType,
    ChatbotResponse,
)


class TestCustomerModels:
    """Test customer-related models."""
    
    def test_customer_profile_creation(self):
        """Test creating a valid CustomerProfile."""
        profile = CustomerProfile(
            customer_id="cust_123",
            age=30,
            location="Manila",
            transaction_frequency=50,
            average_transaction_value=1500.0,
            merchant_categories=["Shopping", "Dining"],
            total_spend=75000.0,
            account_age_days=365,
            preferred_payment_methods=["Credit Card", "E-wallet"],
            last_transaction_date=datetime.now()
        )
        
        assert profile.customer_id == "cust_123"
        assert profile.age == 30
        assert len(profile.merchant_categories) == 2
    
    def test_customer_profile_age_validation(self):
        """Test age validation."""
        with pytest.raises(ValueError):
            CustomerProfile(
                customer_id="cust_123",
                age=150,  # Invalid age
                location="Manila",
                transaction_frequency=50,
                average_transaction_value=1500.0,
                merchant_categories=["Shopping"],
                total_spend=75000.0,
                account_age_days=365,
                preferred_payment_methods=["Credit Card"],
                last_transaction_date=datetime.now()
            )
    
    def test_transaction_data_creation(self):
        """Test creating a valid TransactionData."""
        transaction = TransactionData(
            transaction_id="txn_123",
            customer_id="cust_123",
            amount=500.0,
            merchant_category="Shopping",
            merchant_name="Store ABC",
            transaction_type="payment",
            timestamp=datetime.now(),
            payment_method="Credit Card",
            location="Manila"
        )
        
        assert transaction.transaction_id == "txn_123"
        assert transaction.amount == 500.0
    
    def test_transaction_type_validation(self):
        """Test transaction type validation."""
        with pytest.raises(ValueError):
            TransactionData(
                transaction_id="txn_123",
                customer_id="cust_123",
                amount=500.0,
                merchant_category="Shopping",
                merchant_name="Store ABC",
                transaction_type="invalid_type",  # Invalid type
                timestamp=datetime.now(),
                payment_method="Credit Card",
                location="Manila"
            )


class TestSegmentModels:
    """Test segmentation-related models."""
    
    def test_segment_creation(self):
        """Test creating a valid Segment."""
        segment = Segment(
            segment_id="seg_123",
            name="High-Value Shoppers",
            description="Customers with high transaction values",
            characteristics={"avg_age": 35, "location": "Metro Manila"},
            cluster_id=0,
            centroid=[1.5, 2.3, -0.8],
            size=1500,
            average_transaction_value=3000.0,
            transaction_frequency=45.0,
            top_merchant_categories=["Shopping", "Dining"],
            differentiating_factors=["High spending", "Frequent transactions"],
            pca_component_contributions={0: 0.45, 1: 0.35}
        )
        
        assert segment.segment_id == "seg_123"
        assert segment.size == 1500
        assert len(segment.centroid) == 3
    
    def test_customer_segment_assignment_creation(self):
        """Test creating a valid CustomerSegmentAssignment."""
        assignment = CustomerSegmentAssignment(
            assignment_id="assign_123",
            customer_id="cust_123",
            segment_id="seg_123",
            confidence_score=0.85,
            distance_to_centroid=0.5,
            explanation="Customer has high transaction frequency"
        )
        
        assert assignment.confidence_score == 0.85
        assert 0.0 <= assignment.confidence_score <= 1.0


class TestMLModels:
    """Test ML-related models."""
    
    def test_pca_result_creation(self):
        """Test creating a valid PCAResult."""
        pca_result = PCAResult(
            transformed_data=[[1.5, 2.3], [0.8, -1.2]],
            explained_variance=[2.5, 1.8],
            explained_variance_ratio=[0.6, 0.4],
            components=[[0.7, 0.3], [0.3, 0.7]],
            feature_names=["age", "transaction_frequency"],
            n_components=2
        )
        
        assert pca_result.n_components == 2
        assert len(pca_result.explained_variance_ratio) == 2
    
    def test_clustering_result_creation(self):
        """Test creating a valid ClusteringResult."""
        clustering_result = ClusteringResult(
            cluster_labels=[0, 1, 0, 2, 1],
            centroids=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            inertia=150.5,
            n_clusters=3,
            silhouette_score=0.65
        )
        
        assert clustering_result.n_clusters == 3
        assert -1.0 <= clustering_result.silhouette_score <= 1.0


class TestCampaignModels:
    """Test campaign-related models."""
    
    def test_ad_content_creation(self):
        """Test creating a valid AdContent."""
        ad = AdContent(
            ad_id="ad_123",
            segment_id="seg_123",
            campaign_id="camp_123",
            format=AdFormat.SHORT,
            content="Get 20% cashback!",
            variation_number=1,
            use_case="cashback"
        )
        
        assert ad.format == AdFormat.SHORT
        assert len(ad.content) <= 50  # SHORT format limit
    
    def test_ad_content_length_validation(self):
        """Test ad content length validation."""
        with pytest.raises(ValueError):
            AdContent(
                ad_id="ad_123",
                segment_id="seg_123",
                campaign_id="camp_123",
                format=AdFormat.SHORT,
                content="This is a very long ad content that exceeds the 50 character limit for SHORT format",
                variation_number=1,
                use_case="cashback"
            )
    
    def test_campaign_creation(self):
        """Test creating a valid Campaign."""
        campaign = Campaign(
            campaign_id="camp_123",
            name="Summer Promo",
            target_segment_ids=["seg_123", "seg_456"],
            ad_content_ids=["ad_123", "ad_456"],
            estimated_reach=5000
        )
        
        assert campaign.status == CampaignStatus.DRAFT
        assert len(campaign.target_segment_ids) == 2


class TestLLMModels:
    """Test LLM-related models."""
    
    def test_llm_parameters_creation(self):
        """Test creating valid LLMParameters."""
        params = LLMParameters(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        
        assert 0.0 <= params.temperature <= 2.0
        assert params.max_tokens > 0
    
    def test_llm_configuration_creation(self):
        """Test creating a valid LLMConfiguration."""
        config = LLMConfiguration(
            config_id="config_123",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            api_key="sk-test-key"
        )
        
        assert config.provider == LLMProvider.OPENAI
        assert config.is_active is True


class TestChatbotModels:
    """Test chatbot-related models."""
    
    def test_chat_message_creation(self):
        """Test creating a valid ChatMessage."""
        message = ChatMessage(
            message_id="msg_123",
            role=MessageRole.USER,
            content="Tell me about segment A"
        )
        
        assert message.role == MessageRole.USER
        assert len(message.content) > 0
    
    def test_query_intent_creation(self):
        """Test creating a valid QueryIntent."""
        intent = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={"segment_id": "seg_123"},
            confidence=0.95
        )
        
        assert intent.intent_type == QueryType.SEGMENT_INFO
        assert 0.0 <= intent.confidence <= 1.0
    
    def test_chatbot_response_creation(self):
        """Test creating a valid ChatbotResponse."""
        response = ChatbotResponse(
            response_id="resp_123",
            session_id="session_123",
            text="Segment A has 1500 customers",
            response_time_ms=250
        )
        
        assert response.response_time_ms >= 0
        assert len(response.text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
