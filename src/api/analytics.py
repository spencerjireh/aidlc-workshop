"""Analytics endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_segment_repo, get_campaign_repo
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository
from src.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


def get_analytics_service(
    segment_repo: SegmentDataRepository = Depends(get_segment_repo),
    campaign_repo: CampaignDataRepository = Depends(get_campaign_repo),
) -> AnalyticsService:
    return AnalyticsService(segment_repo=segment_repo, campaign_repo=campaign_repo)


@router.get("/segments/distribution")
def get_segment_distribution(
    service: AnalyticsService = Depends(get_analytics_service),
):
    """Get segment distribution statistics."""
    return service.get_segment_distribution()


@router.get("/campaigns/{campaign_id}/performance")
def get_campaign_performance(
    campaign_id: str,
    service: AnalyticsService = Depends(get_analytics_service),
):
    """Get campaign performance metrics."""
    try:
        return service.get_campaign_performance(campaign_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
