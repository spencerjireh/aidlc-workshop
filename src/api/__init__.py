"""FastAPI REST API endpoints."""

from fastapi import FastAPI

from src.api.customers import router as customers_router
from src.api.segments import router as segments_router
from src.api.ads import router as ads_router
from src.api.campaigns import router as campaigns_router
from src.api.analytics import router as analytics_router
from src.api.chatbot import router as chatbot_router
from src.api.llm_config import router as llm_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Customer Segmentation Ads",
        description="ML-powered customer segmentation advertising system using PCA, K-Means, and LLMs",
        version="0.1.0",
    )

    prefix = "/api/v1"
    app.include_router(customers_router, prefix=prefix)
    app.include_router(segments_router, prefix=prefix)
    app.include_router(ads_router, prefix=prefix)
    app.include_router(campaigns_router, prefix=prefix)
    app.include_router(analytics_router, prefix=prefix)
    app.include_router(chatbot_router, prefix=prefix)
    app.include_router(llm_router, prefix=prefix)

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    return app
