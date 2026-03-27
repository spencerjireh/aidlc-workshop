"""Query chatbot service for conversational analytics."""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.engines.llm_engine import LLMEngine
from src.models.chatbot import (
    ChatbotResponse,
    ChatMessage,
    ConversationContext,
    MessageRole,
    QueryIntent,
    QueryType,
)
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository

logger = logging.getLogger(__name__)


class QueryChatbotService:
    """Conversational interface for querying segmentation data."""

    def __init__(
        self,
        llm_engine: LLMEngine,
        segment_repo: SegmentDataRepository,
        customer_repo: CustomerDataRepository,
        campaign_repo: CampaignDataRepository,
    ):
        self.llm_engine = llm_engine
        self.segment_repo = segment_repo
        self.customer_repo = customer_repo
        self.campaign_repo = campaign_repo
        self._sessions: Dict[str, ConversationContext] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        session_id: str,
        user_id: str = "anonymous",
    ) -> ChatbotResponse:
        """Process a natural-language query and return a response."""
        start_time = time.time()
        context = self.get_conversation_context(session_id, user_id)

        # Record user message
        user_msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            role=MessageRole.USER,
            content=query,
        )
        context.conversation_history.append(user_msg)
        context.last_interaction = datetime.utcnow()

        # Interpret intent
        intent = self.llm_engine.interpret_query(query, context)
        user_msg.query_intent = intent

        # Low confidence -- suggest alternatives
        if intent.confidence < 0.5:
            suggestions = self.suggest_alternative_queries(query)
            elapsed_ms = int((time.time() - start_time) * 1000)
            response = ChatbotResponse(
                response_id=f"response_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                text=(
                    "I'm not sure I understand your question. "
                    "Could you try rephrasing it?"
                ),
                suggested_followups=suggestions,
                response_time_ms=elapsed_ms,
            )
            self._record_assistant_message(context, response)
            return response

        # Retrieve data
        data = self._retrieve_data_for_intent(intent)

        # Generate response via LLM
        response = self.llm_engine.generate_response(intent, data, context)

        self._record_assistant_message(context, response)
        return response

    def get_conversation_context(
        self,
        session_id: str,
        user_id: str = "anonymous",
    ) -> ConversationContext:
        """Return existing session context or create a new one."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationContext(
                session_id=session_id,
                user_id=user_id,
            )
        return self._sessions[session_id]

    def suggest_alternative_queries(self, failed_query: str) -> List[str]:
        """Suggest alternative questions the user can ask."""
        segments = self.segment_repo.list_segments()
        suggestions: List[str] = []

        if segments:
            first = segments[0].name
            suggestions.append(f"Tell me about the {first} segment")
            suggestions.append(f"How many customers are in the {first} segment?")
            if len(segments) >= 2:
                second = segments[1].name
                suggestions.append(f"Compare {first} and {second} segments")
            suggestions.append(f"What are the top categories for {first}?")
        else:
            suggestions.append("How many customer segments exist?")
            suggestions.append("Show me the segment distribution")
            suggestions.append("What campaigns are active?")

        return suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retrieve_data_for_intent(self, intent: QueryIntent) -> Dict[str, Any]:
        """Fetch the right data based on intent type."""
        data: Dict[str, Any] = {}
        entities = intent.entities

        if intent.intent_type == QueryType.SEGMENT_INFO:
            segment = self._find_segment(entities)
            if segment:
                data = segment.model_dump()

        elif intent.intent_type == QueryType.COMPARISON:
            seg_a = self._find_segment(entities, key="segment_a")
            seg_b = self._find_segment(entities, key="segment_b")
            if seg_a:
                data["segment_a"] = seg_a.model_dump()
            if seg_b:
                data["segment_b"] = seg_b.model_dump()

        elif intent.intent_type == QueryType.PERFORMANCE:
            campaign_id = entities.get("campaign_id") or entities.get("campaign")
            if campaign_id:
                campaign = self.campaign_repo.get_campaign(str(campaign_id))
                if campaign:
                    data = campaign.model_dump()

        elif intent.intent_type == QueryType.TREND:
            segment = self._find_segment(entities)
            if segment:
                data["versions"] = self.segment_repo.get_segment_version_history(
                    segment.segment_id
                )

        elif intent.intent_type == QueryType.CUSTOMER_COUNT:
            segment = self._find_segment(entities)
            if segment:
                data = {
                    "segment_name": segment.name,
                    "customer_count": self.segment_repo.get_segment_size(
                        segment.segment_id
                    ),
                }

        elif intent.intent_type == QueryType.TOP_CATEGORIES:
            segment = self._find_segment(entities)
            if segment:
                data = {
                    "segment_name": segment.name,
                    "top_categories": segment.top_merchant_categories,
                }

        return data

    def _find_segment(
        self, entities: Dict[str, Any], key: str = "segment"
    ) -> Optional[Any]:
        """Look up a segment by id or name from entities."""
        value = entities.get(key) or entities.get("segment_id") or entities.get("segment_name")
        if not value:
            # Fall back to first segment
            segments = self.segment_repo.list_segments()
            return segments[0] if segments else None

        value_str = str(value)

        # Try direct ID lookup
        seg = self.segment_repo.get_segment(value_str)
        if seg:
            return seg

        # Try name match
        for seg in self.segment_repo.list_segments():
            if seg.name.lower() == value_str.lower():
                return seg
            if seg.segment_id == value_str:
                return seg

        return None

    def _record_assistant_message(
        self, context: ConversationContext, response: ChatbotResponse
    ) -> None:
        assistant_msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            role=MessageRole.ASSISTANT,
            content=response.text,
        )
        context.conversation_history.append(assistant_msg)
        context.last_interaction = datetime.utcnow()
