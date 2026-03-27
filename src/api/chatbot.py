"""Chatbot endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_chatbot_service
from src.api.schemas import ChatbotQueryRequest
from src.services.query_chatbot_service import QueryChatbotService

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router.post("/query")
def process_query(
    request: ChatbotQueryRequest,
    service: QueryChatbotService = Depends(get_chatbot_service),
):
    """Process a chatbot query."""
    try:
        response = service.process_query(
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/context")
def get_conversation_context(
    session_id: str,
    user_id: str = "anonymous",
    service: QueryChatbotService = Depends(get_chatbot_service),
):
    """Get conversation context for a session."""
    context = service.get_conversation_context(session_id, user_id)
    return context.model_dump()
