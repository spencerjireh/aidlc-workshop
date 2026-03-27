"""Integration tests for FastAPI REST API endpoints (Task 14.9)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.api import create_app
from src.api.dependencies import (
    _customer_repo, _segment_repo, _campaign_repo,
    get_customer_repo, get_segment_repo, get_campaign_repo,
)
from src.models.customer import CustomerProfile
from src.models.segment import Segment, CustomerSegmentAssignment, ContributingFactor
from src.models.campaign import Campaign, AdContent, AdFormat, CampaignStatus
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def customer_repo():
    return CustomerDataRepository()


@pytest.fixture
def segment_repo():
    return SegmentDataRepository()


@pytest.fixture
def campaign_repo():
    return CampaignDataRepository()


class TestHealthCheck:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestCustomerEndpoints:
    def test_get_customer_not_found(self, client):
        response = client.get("/api/v1/customers/nonexistent")
        assert response.status_code == 404

    def test_ingest_customers(self, client):
        customers = [
            {
                "customer_id": "test_001",
                "age": 30,
                "location": "Manila",
                "transaction_frequency": 50,
                "average_transaction_value": 1500.0,
                "merchant_categories": ["Food", "Transport"],
                "total_spend": 75000.0,
                "account_age_days": 365,
                "preferred_payment_methods": ["Credit Card"],
                "last_transaction_date": datetime.utcnow().isoformat(),
            }
        ]
        response = client.post(
            "/api/v1/customers/ingest",
            json={"customers": customers, "format": "json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["successful"] >= 0
        assert data["total_records"] >= 0

    def test_get_customer_assignment_not_found(self, client):
        response = client.get("/api/v1/customers/nonexistent/assignment")
        assert response.status_code == 404

    def test_get_customer_explanation_not_found(self, client):
        response = client.get("/api/v1/customers/nonexistent/explanation")
        assert response.status_code == 404


class TestSegmentEndpoints:
    def test_list_segments_empty(self, client):
        response = client.get("/api/v1/segments")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_segment_not_found(self, client):
        response = client.get("/api/v1/segments/nonexistent")
        assert response.status_code == 404

    def test_get_segment_customers_not_found(self, client):
        response = client.get("/api/v1/segments/nonexistent/customers")
        assert response.status_code == 404


class TestCampaignEndpoints:
    def test_list_campaigns_empty(self, client):
        response = client.get("/api/v1/campaigns")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_campaign_not_found(self, client):
        response = client.get("/api/v1/campaigns/nonexistent")
        assert response.status_code == 404

    def test_activate_nonexistent_campaign(self, client):
        response = client.post("/api/v1/campaigns/nonexistent/activate")
        assert response.status_code in (400, 500)

    def test_get_campaign_reach_not_found(self, client):
        response = client.get("/api/v1/campaigns/nonexistent/reach")
        assert response.status_code == 404


class TestAdEndpoints:
    def test_get_ad_not_found(self, client):
        response = client.get("/api/v1/ads/nonexistent")
        assert response.status_code == 404


class TestAnalyticsEndpoints:
    def test_get_segment_distribution(self, client):
        response = client.get("/api/v1/analytics/segments/distribution")
        assert response.status_code == 200

    def test_get_campaign_performance_not_found(self, client):
        response = client.get("/api/v1/analytics/campaigns/nonexistent/performance")
        assert response.status_code == 404


class TestChatbotEndpoints:
    def test_get_conversation_context(self, client):
        response = client.get("/api/v1/chatbot/sessions/test_session/context")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"


class TestLLMConfigEndpoints:
    def test_list_providers(self, client):
        response = client.get("/api/v1/llm/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) == 3

    def test_configure_llm(self, client):
        response = client.post(
            "/api/v1/llm/configure",
            json={
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "test-key",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "configured"
        assert data["provider"] == "openai"

    def test_validate_credentials(self, client):
        response = client.post(
            "/api/v1/llm/validate",
            json={
                "provider": "openai",
                "api_key": "invalid-key",
                "model_name": "gpt-4",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data


class TestResponseFormats:
    def test_404_returns_detail(self, client):
        response = client.get("/api/v1/customers/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_invalid_request_body(self, client):
        response = client.post(
            "/api/v1/customers/ingest",
            json={"invalid": "body"},
        )
        assert response.status_code == 422  # Validation error
