"""Ad generator service for creating personalized ad content."""

import logging
from typing import List

from src.engines.llm_engine import LLMEngine
from src.models.campaign import AdContent, AdFormat
from src.models.segment import SegmentProfile
from src.models.service_models import ContentValidationResult
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.segment_repository import SegmentDataRepository

logger = logging.getLogger(__name__)

BLOCKED_TERMS = [
    "guaranteed",
    "risk-free",
    "free money",
    "get rich",
    "no risk",
    "100% safe",
    "act now or lose",
    "limited time only",
]


class AdGeneratorService:
    """Service for generating personalized ad content for customer segments."""

    def __init__(
        self,
        llm_engine: LLMEngine,
        segment_repo: SegmentDataRepository,
        campaign_repo: CampaignDataRepository,
    ):
        self.llm_engine = llm_engine
        self.segment_repo = segment_repo
        self.campaign_repo = campaign_repo

    def create_campaign_ads(
        self,
        segment_id: str,
        campaign_id: str,
        campaign_brief: str,
        formats: List[AdFormat],
        num_variations: int = 3,
    ) -> List[AdContent]:
        """Generate ad content for a campaign targeting a specific segment.

        Creates ``num_variations`` (minimum 3) per format for A/B testing.
        """
        if num_variations < 3:
            raise ValueError("num_variations must be at least 3 for A/B testing")

        segment = self.segment_repo.get_segment(segment_id)
        if segment is None:
            raise ValueError(f"Segment {segment_id} not found")

        profile = SegmentProfile(
            segment_id=segment.segment_id,
            name=segment.name,
            description=segment.description,
            differentiating_factors=segment.differentiating_factors,
        )

        all_ads: List[AdContent] = []
        for ad_format in formats:
            ads = self.llm_engine.generate_ad_content(
                profile, campaign_brief, ad_format, num_variations
            )
            for ad in ads:
                # Override the empty campaign_id set by LLMEngine
                ad.campaign_id = campaign_id

                validation = self.validate_ad_content(ad.content)
                if not validation.is_approved:
                    logger.warning(
                        "Ad %s rejected: %s -- skipping",
                        ad.ad_id,
                        validation.issues,
                    )
                    continue

                self.campaign_repo.create_ad_content(ad)
                all_ads.append(ad)

        return all_ads

    def validate_ad_content(self, content: str) -> ContentValidationResult:
        """Check ad content for empty or inappropriate text."""
        issues: List[str] = []

        if not content or not content.strip():
            issues.append("Ad content is empty")
            return ContentValidationResult(is_approved=False, issues=issues)

        lower = content.lower()
        for term in BLOCKED_TERMS:
            if term in lower:
                issues.append(f"Contains blocked term: '{term}'")

        return ContentValidationResult(is_approved=len(issues) == 0, issues=issues)
