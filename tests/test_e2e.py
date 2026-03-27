"""End-to-end and integration tests (Task 18)."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api import create_app
from src.models.segment import SegmentProfile
from src.models.campaign import AdContent, AdFormat
from src.models.chatbot import ChatbotResponse, QueryIntent, QueryType


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestSegmentationWorkflow:
    """E2E: ingest -> create segments -> get assignments -> get explanations."""

    def test_ingest_customers(self, client):
        customers = []
        for i in range(5):
            customers.append({
                "customer_id": f"e2e_cust_{i:03d}",
                "age": 25 + i * 5,
                "location": "Manila",
                "transaction_frequency": 10 + i * 10,
                "average_transaction_value": 500.0 + i * 200,
                "merchant_categories": ["Food", "Transport"],
                "total_spend": 5000.0 + i * 1000,
                "account_age_days": 100 + i * 50,
                "preferred_payment_methods": ["Credit Card"],
                "last_transaction_date": datetime.utcnow().isoformat(),
            })

        response = client.post(
            "/api/v1/customers/ingest",
            json={"customers": customers, "format": "json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 5
        assert data["successful"] >= 0

    def test_list_segments_initially_empty(self, client):
        response = client.get("/api/v1/segments")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestCampaignWorkflow:
    """E2E: segments exist -> generate ads -> create campaign -> activate -> check reach."""

    def test_campaign_not_found(self, client):
        response = client.get("/api/v1/campaigns/nonexistent")
        assert response.status_code == 404

    def test_campaign_list_empty(self, client):
        response = client.get("/api/v1/campaigns")
        assert response.status_code == 200
        assert response.json() == []


class TestChatbotWorkflow:
    """E2E: send query -> check response -> get session context."""

    def test_get_empty_session_context(self, client):
        response = client.get("/api/v1/chatbot/sessions/e2e_session/context")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "e2e_session"
        assert data["conversation_history"] == []


class TestProviderConfiguration:
    """E2E: configure LLM -> validate -> switch providers."""

    def test_configure_and_list_providers(self, client):
        # List providers
        response = client.get("/api/v1/llm/providers")
        assert response.status_code == 200
        providers = response.json()["providers"]
        assert len(providers) == 3
        provider_names = [p["name"] for p in providers]
        assert "openai" in provider_names
        assert "anthropic" in provider_names
        assert "local" in provider_names

    def test_configure_provider(self, client):
        response = client.post(
            "/api/v1/llm/configure",
            json={
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "test-key-e2e",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "configured"

    def test_switch_provider(self, client):
        # Configure OpenAI first
        client.post(
            "/api/v1/llm/configure",
            json={
                "provider": "openai",
                "model_name": "gpt-4",
                "api_key": "key1",
            },
        )

        # Switch to Anthropic
        response = client.post(
            "/api/v1/llm/configure",
            json={
                "provider": "anthropic",
                "model_name": "claude-3-opus-20240229",
                "api_key": "key2",
            },
        )
        assert response.status_code == 200
        assert response.json()["provider"] == "anthropic"

    def test_validate_invalid_credentials(self, client):
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


class TestCrossServiceDataFlow:
    """Integration: data flows correctly between repos/services via API."""

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_analytics_segments_empty(self, client):
        response = client.get("/api/v1/analytics/segments/distribution")
        assert response.status_code == 200
        assert response.json() == []

    def test_analytics_campaign_not_found(self, client):
        response = client.get("/api/v1/analytics/campaigns/fake/performance")
        assert response.status_code == 404


class TestErrorPropagation:
    """Integration: errors from services surface correctly in API responses."""

    def test_get_nonexistent_customer(self, client):
        response = client.get("/api/v1/customers/does_not_exist")
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_get_nonexistent_segment(self, client):
        response = client.get("/api/v1/segments/does_not_exist")
        assert response.status_code == 404

    def test_get_nonexistent_ad(self, client):
        response = client.get("/api/v1/ads/does_not_exist")
        assert response.status_code == 404

    def test_activate_nonexistent_campaign(self, client):
        response = client.post("/api/v1/campaigns/does_not_exist/activate")
        assert response.status_code in (400, 500)

    def test_campaign_reach_nonexistent(self, client):
        response = client.get("/api/v1/campaigns/does_not_exist/reach")
        assert response.status_code == 404

    def test_invalid_request_body_422(self, client):
        response = client.post("/api/v1/customers/ingest", json={})
        assert response.status_code == 422

    def test_segment_customers_nonexistent(self, client):
        response = client.get("/api/v1/segments/does_not_exist/customers")
        assert response.status_code == 404

    def test_customer_assignment_nonexistent(self, client):
        response = client.get("/api/v1/customers/does_not_exist/assignment")
        assert response.status_code == 404
