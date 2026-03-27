"""Tests for QueryChatbotService."""

import pytest
from unittest.mock import MagicMock

from src.engines.llm_engine import LLMEngine
from src.models.chatbot import (
    ChatbotResponse,
    ConversationContext,
    QueryIntent,
    QueryType,
)
from src.models.segment import Segment
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.query_chatbot_service import QueryChatbotService


def _make_segment(segment_id="segment_0", name="Young Spenders"):
    return Segment(
        segment_id=segment_id,
        name=name,
        description="Young frequent spenders",
        characteristics={"average_age": 25.0},
        cluster_id=0,
        centroid=[0.1, 0.2],
        size=200,
        average_transaction_value=500.0,
        transaction_frequency=30.0,
        top_merchant_categories=["Food", "Shopping"],
        differentiating_factors=["High frequency"],
        pca_component_contributions={0: 0.6},
    )


def _make_chatbot_response(session_id="sess_1", text="Here is the info."):
    return ChatbotResponse(
        response_id="resp_1",
        session_id=session_id,
        text=text,
        data={"key": "value"},
        suggested_followups=["What else?", "Tell me more"],
        response_time_ms=100,
    )


def _build_service(llm=None, segment_repo=None, customer_repo=None, campaign_repo=None):
    return QueryChatbotService(
        llm_engine=llm or MagicMock(spec=LLMEngine),
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        customer_repo=customer_repo or MagicMock(spec=CustomerDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


class TestProcessQuery:
    def test_segment_info_query(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.list_segments.return_value = [_make_segment()]
        segment_repo.get_segment.return_value = _make_segment()

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={"segment": "segment_0"},
            confidence=0.9,
        )
        llm.generate_response.return_value = _make_chatbot_response()

        service = _build_service(llm=llm, segment_repo=segment_repo)
        response = service.process_query("Tell me about segment 0", "sess_1")

        assert isinstance(response, ChatbotResponse)
        assert response.text == "Here is the info."
        llm.interpret_query.assert_called_once()
        llm.generate_response.assert_called_once()

    def test_customer_count_query(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.get_segment_size.return_value = 200

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.CUSTOMER_COUNT,
            entities={"segment": "segment_0"},
            confidence=0.95,
        )
        llm.generate_response.return_value = _make_chatbot_response(
            text="There are 200 customers."
        )

        service = _build_service(llm=llm, segment_repo=segment_repo)
        response = service.process_query("How many customers?", "sess_1")
        assert "200" in response.text

    def test_comparison_query(self):
        seg_a = _make_segment("seg_a", "Young Spenders")
        seg_b = _make_segment("seg_b", "Bill Payers")
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.side_effect = lambda sid: (
            seg_a if sid == "seg_a" else seg_b if sid == "seg_b" else None
        )
        segment_repo.list_segments.return_value = [seg_a, seg_b]

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.COMPARISON,
            entities={"segment_a": "seg_a", "segment_b": "seg_b"},
            confidence=0.85,
        )
        llm.generate_response.return_value = _make_chatbot_response(
            text="Segment A vs B comparison."
        )

        service = _build_service(llm=llm, segment_repo=segment_repo)
        response = service.process_query("Compare A and B", "sess_1")
        assert isinstance(response, ChatbotResponse)

    def test_low_confidence_suggests_alternatives(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.list_segments.return_value = [
            _make_segment("seg_1", "Young Spenders"),
            _make_segment("seg_2", "Bill Payers"),
        ]

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={},
            confidence=0.3,
        )

        service = _build_service(llm=llm, segment_repo=segment_repo)
        response = service.process_query("asdf gibberish", "sess_1")

        assert "not sure" in response.text.lower() or "rephras" in response.text.lower()
        assert len(response.suggested_followups) >= 3
        llm.generate_response.assert_not_called()


class TestConversationContext:
    def test_creates_new_session(self):
        service = _build_service()
        ctx = service.get_conversation_context("new_session", "user1")
        assert isinstance(ctx, ConversationContext)
        assert ctx.session_id == "new_session"
        assert ctx.user_id == "user1"
        assert len(ctx.conversation_history) == 0

    def test_returns_existing_session(self):
        service = _build_service()
        ctx1 = service.get_conversation_context("sess_1", "user1")
        ctx2 = service.get_conversation_context("sess_1", "user1")
        assert ctx1 is ctx2

    def test_history_persisted_across_queries(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.list_segments.return_value = [_make_segment()]

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={"segment": "segment_0"},
            confidence=0.9,
        )
        llm.generate_response.return_value = _make_chatbot_response()

        service = _build_service(llm=llm, segment_repo=segment_repo)

        service.process_query("First question", "sess_1")
        service.process_query("Second question", "sess_1")

        ctx = service.get_conversation_context("sess_1")
        # 2 user messages + 2 assistant messages
        assert len(ctx.conversation_history) == 4


class TestSuggestAlternatives:
    def test_returns_suggestions_with_segments(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.list_segments.return_value = [
            _make_segment("seg_1", "Young Spenders"),
            _make_segment("seg_2", "Bill Payers"),
        ]

        service = _build_service(segment_repo=segment_repo)
        suggestions = service.suggest_alternative_queries("bad query")
        assert len(suggestions) >= 3
        assert any("Young Spenders" in s for s in suggestions)

    def test_returns_suggestions_without_segments(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.list_segments.return_value = []

        service = _build_service(segment_repo=segment_repo)
        suggestions = service.suggest_alternative_queries("bad query")
        assert len(suggestions) >= 3
