"""Campaign management endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_campaign_repo, get_targeting_engine
from src.api.schemas import CampaignCreateRequest
from src.repositories.campaign_repository import CampaignDataRepository
from src.services.targeting_engine import TargetingEngine

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.post("/create")
def create_campaign(
    request: CampaignCreateRequest,
    engine: TargetingEngine = Depends(get_targeting_engine),
):
    """Create a campaign targeting specific segments."""
    try:
        campaign = engine.create_campaign(
            campaign_name=request.name,
            target_segment_ids=request.target_segment_ids,
            ad_content_ids=request.ad_content_ids,
        )
        return campaign.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
def list_campaigns(
    repo: CampaignDataRepository = Depends(get_campaign_repo),
):
    """List all campaigns."""
    campaigns = repo.list_campaigns()
    return [c.model_dump() for c in campaigns]


@router.get("/{campaign_id}")
def get_campaign(
    campaign_id: str,
    repo: CampaignDataRepository = Depends(get_campaign_repo),
):
    """Get campaign details."""
    campaign = repo.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign.model_dump()


@router.post("/{campaign_id}/activate")
def activate_campaign(
    campaign_id: str,
    engine: TargetingEngine = Depends(get_targeting_engine),
):
    """Activate a campaign."""
    try:
        result = engine.activate_campaign(campaign_id)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/reach")
def get_campaign_reach(
    campaign_id: str,
    repo: CampaignDataRepository = Depends(get_campaign_repo),
    engine: TargetingEngine = Depends(get_targeting_engine),
):
    """Calculate estimated reach for a campaign."""
    campaign = repo.get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    reach = engine.calculate_reach(campaign.target_segment_ids)
    return reach.model_dump()
