"""Shared dependencies and service instances for API endpoints."""

from src.config import settings, get_llm_config
from src.engines.pca_engine import PCAEngine
from src.engines.kmeans_engine import KMeansEngine
from src.engines.llm_engine import LLMEngine
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository
from src.services.segmentation_service import SegmentationService
from src.services.ad_generator_service import AdGeneratorService
from src.services.targeting_engine import TargetingEngine
from src.services.query_chatbot_service import QueryChatbotService
from src.models.llm import LLMConfiguration, LLMParameters

# Singleton instances (in-memory POC)
_customer_repo = CustomerDataRepository()
_segment_repo = SegmentDataRepository()
_campaign_repo = CampaignDataRepository()

_pca_engine = PCAEngine()
_kmeans_engine = KMeansEngine()

# LLM engine (initialized lazily)
_llm_engine = None
_llm_config = None


def _get_llm_engine() -> LLMEngine:
    global _llm_engine, _llm_config
    if _llm_engine is None:
        provider_config = get_llm_config()
        _llm_config = LLMConfiguration(
            config_id="default",
            provider=provider_config.provider,
            model_name=provider_config.model_name,
            api_key=provider_config.api_key or "",
            api_endpoint=provider_config.api_endpoint,
            parameters=LLMParameters(
                temperature=provider_config.temperature,
                max_tokens=provider_config.max_tokens,
            ),
        )
        _llm_engine = LLMEngine(configuration=_llm_config)
    return _llm_engine


def get_customer_repo() -> CustomerDataRepository:
    return _customer_repo


def get_segment_repo() -> SegmentDataRepository:
    return _segment_repo


def get_campaign_repo() -> CampaignDataRepository:
    return _campaign_repo


def get_segmentation_service() -> SegmentationService:
    return SegmentationService(
        pca_engine=_pca_engine,
        kmeans_engine=_kmeans_engine,
        llm_engine=_get_llm_engine(),
        customer_repo=_customer_repo,
        segment_repo=_segment_repo,
    )


def get_ad_generator_service() -> AdGeneratorService:
    return AdGeneratorService(
        llm_engine=_get_llm_engine(),
        segment_repo=_segment_repo,
        campaign_repo=_campaign_repo,
    )


def get_targeting_engine() -> TargetingEngine:
    return TargetingEngine(
        segment_repo=_segment_repo,
        campaign_repo=_campaign_repo,
    )


def get_chatbot_service() -> QueryChatbotService:
    return QueryChatbotService(
        llm_engine=_get_llm_engine(),
        segment_repo=_segment_repo,
        customer_repo=_customer_repo,
        campaign_repo=_campaign_repo,
    )


def reconfigure_llm(config: LLMConfiguration) -> None:
    """Reconfigure the LLM engine with new settings."""
    global _llm_engine, _llm_config
    _llm_config = config
    _llm_engine = LLMEngine(configuration=config)
