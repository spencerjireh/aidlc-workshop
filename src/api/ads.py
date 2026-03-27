"""Ad generation endpoints."""

import uuid
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_ad_generator_service, get_campaign_repo
from src.api.schemas import AdGenerateRequest
from src.repositories.campaign_repository import CampaignDataRepository
from src.services.ad_generator_service import AdGeneratorService

router = APIRouter(prefix="/ads", tags=["ads"])


@router.post("/generate")
def generate_ads(
    request: AdGenerateRequest,
    service: AdGeneratorService = Depends(get_ad_generator_service),
):
    """Generate ad content for a segment."""
    campaign_id = f"campaign_{uuid.uuid4().hex[:8]}"
    try:
        ads = service.create_campaign_ads(
            segment_id=request.segment_id,
            campaign_id=campaign_id,
            campaign_brief=request.campaign_brief,
            formats=request.formats,
            num_variations=request.num_variations,
        )
        return [ad.model_dump() for ad in ads]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ad_id}")
def get_ad(
    ad_id: str,
    repo: CampaignDataRepository = Depends(get_campaign_repo),
):
    """Get ad content details."""
    ad = repo.get_ad_content(ad_id)
    if not ad:
        raise HTTPException(status_code=404, detail="Ad not found")
    return ad.model_dump()
