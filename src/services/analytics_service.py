"""Analytics service for segment distribution and campaign performance."""

from typing import Dict, List

from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository


class AnalyticsService:
    """Service for analytics: segment distribution and campaign performance."""

    def __init__(
        self,
        segment_repo: SegmentDataRepository,
        campaign_repo: CampaignDataRepository,
    ):
        self.segment_repo = segment_repo
        self.campaign_repo = campaign_repo

    def get_segment_distribution(self) -> List[Dict]:
        """
        Get segment distribution statistics.

        Returns list of dicts with segment_id, name, customer_count, percentage.
        Percentages sum to 100% (within float tolerance).
        """
        segments = self.segment_repo.list_segments()
        if not segments:
            return []

        total_customers = sum(s.size for s in segments)
        if total_customers == 0:
            return [
                {
                    "segment_id": s.segment_id,
                    "name": s.name,
                    "customer_count": 0,
                    "percentage": 0.0,
                }
                for s in segments
            ]

        return [
            {
                "segment_id": s.segment_id,
                "name": s.name,
                "customer_count": s.size,
                "percentage": round((s.size / total_customers) * 100, 2),
            }
            for s in segments
        ]

    def get_campaign_performance(self, campaign_id: str) -> Dict:
        """
        Get campaign performance metrics.

        Returns dict with campaign info and ads with performance metrics.
        Raises ValueError if campaign not found.
        """
        campaign = self.campaign_repo.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign not found: {campaign_id}")

        ads = self.campaign_repo.get_campaign_ads(campaign_id)
        ad_metrics = []
        for ad in ads:
            metrics = self.campaign_repo.get_performance_metrics(ad.ad_id)
            if metrics:
                ad_metrics.append({
                    "ad_id": ad.ad_id,
                    "format": ad.format.value,
                    "impressions": metrics.impressions,
                    "clicks": metrics.clicks,
                    "conversions": metrics.conversions,
                    "click_through_rate": metrics.click_through_rate,
                    "conversion_rate": metrics.conversion_rate,
                })
            else:
                ad_metrics.append({
                    "ad_id": ad.ad_id,
                    "format": ad.format.value,
                    "impressions": 0,
                    "clicks": 0,
                    "conversions": 0,
                    "click_through_rate": 0.0,
                    "conversion_rate": 0.0,
                })

        return {
            "campaign_id": campaign.campaign_id,
            "name": campaign.name,
            "status": campaign.status.value,
            "target_segments": campaign.target_segment_ids,
            "estimated_reach": campaign.estimated_reach,
            "ads": ad_metrics,
        }
