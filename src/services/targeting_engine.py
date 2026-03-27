"""Targeting engine for campaign management and audience estimation."""

import uuid
from datetime import datetime
from typing import List

from src.models.campaign import Campaign, CampaignStatus
from src.models.service_models import CampaignActivationResult, ReachEstimate
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.segment_repository import SegmentDataRepository

MIN_SEGMENT_SIZE = 100


class TargetingEngine:
    """Manages campaign targeting, reach estimation, and activation."""

    def __init__(
        self,
        segment_repo: SegmentDataRepository,
        campaign_repo: CampaignDataRepository,
    ):
        self.segment_repo = segment_repo
        self.campaign_repo = campaign_repo

    def create_campaign(
        self,
        campaign_name: str,
        target_segment_ids: List[str],
        ad_content_ids: List[str],
    ) -> Campaign:
        """Create a DRAFT campaign targeting one or more segments.

        Validates that all segments exist and have at least 100 customers.
        """
        if not target_segment_ids:
            raise ValueError("At least one target segment is required")
        if not ad_content_ids:
            raise ValueError("At least one ad content ID is required")

        for sid in target_segment_ids:
            if not self.segment_repo.segment_exists(sid):
                raise ValueError(f"Segment {sid} does not exist")
            size = self.segment_repo.get_segment_size(sid)
            if size < MIN_SEGMENT_SIZE:
                raise ValueError(
                    f"Segment {sid} has {size} customers, "
                    f"minimum {MIN_SEGMENT_SIZE} required"
                )

        reach = self.calculate_reach(target_segment_ids)

        campaign = Campaign(
            campaign_id=f"campaign_{uuid.uuid4().hex[:8]}",
            name=campaign_name,
            target_segment_ids=target_segment_ids,
            ad_content_ids=ad_content_ids,
            status=CampaignStatus.DRAFT,
            estimated_reach=reach.total_customers,
        )
        self.campaign_repo.create_campaign(campaign)
        return campaign

    def calculate_reach(self, segment_ids: List[str]) -> ReachEstimate:
        """Calculate estimated audience reach for the given segments."""
        breakdown: dict[str, int] = {}
        for sid in segment_ids:
            breakdown[sid] = self.segment_repo.get_segment_size(sid)

        return ReachEstimate(
            total_customers=sum(breakdown.values()),
            segment_breakdown=breakdown,
        )

    def activate_campaign(self, campaign_id: str) -> CampaignActivationResult:
        """Activate a campaign after validation checks."""
        campaign = self.campaign_repo.get_campaign(campaign_id)
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found")

        if not campaign.ad_content_ids:
            raise ValueError("Campaign has no ad content associated")
        if not campaign.target_segment_ids:
            raise ValueError("Campaign has no target segments")

        # Re-validate segment sizes
        for sid in campaign.target_segment_ids:
            size = self.segment_repo.get_segment_size(sid)
            if size < MIN_SEGMENT_SIZE:
                raise ValueError(
                    f"Segment {sid} has {size} customers, "
                    f"minimum {MIN_SEGMENT_SIZE} required"
                )

        campaign.status = CampaignStatus.ACTIVE
        campaign.start_date = datetime.utcnow()
        campaign.updated_at = datetime.utcnow()
        self.campaign_repo.update_campaign(campaign_id, campaign)

        return CampaignActivationResult(
            campaign_id=campaign_id,
            status=CampaignStatus.ACTIVE,
            activated_at=campaign.start_date,
            segments_targeted=campaign.target_segment_ids,
            ads_associated=campaign.ad_content_ids,
        )
