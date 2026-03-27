"""Property-based tests for AnalyticsService.

Uses Hypothesis strategies with mocked repositories to validate invariants
for segment distribution and campaign performance metrics.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.models.campaign import (
    AdContent,
    AdFormat,
    AdPerformanceMetrics,
    Campaign,
    CampaignStatus,
)
from src.models.segment import Segment
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.analytics_service import AnalyticsService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(cluster_id: int = 0, size: int = 100, segment_id: str = None) -> Segment:
    return Segment(
        segment_id=segment_id or f"segment_{cluster_id}",
        name=f"Segment {cluster_id}",
        description=f"Description for segment {cluster_id}",
        characteristics={"average_age": 30.0},
        cluster_id=cluster_id,
        centroid=[0.1, 0.2],
        size=size,
        average_transaction_value=500.0,
        transaction_frequency=20.0,
        top_merchant_categories=["Food"],
        differentiating_factors=["High frequency"],
        pca_component_contributions={0: 0.6},
    )


def _build_analytics_service(segment_repo=None, campaign_repo=None) -> AnalyticsService:
    return AnalyticsService(
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


# ===========================================================================
# Analytics Service property tests
# ===========================================================================


class TestAnalyticsProperties:
    """Property-based tests for AnalyticsService."""

    # -- Property 19: Segment distribution consistency --------------------
    @settings(max_examples=50)
    @given(
        sizes=st.lists(
            st.integers(min_value=1, max_value=10000),
            min_size=1,
            max_size=8,
        )
    )
    def test_segment_distribution_consistency(self, sizes):
        """Property 19 -- Percentages sum to 100.

        Requirements: get_segment_distribution returns percentage values
        that sum to 100 (within floating-point tolerance of 1.0).
        """
        segments = [
            _make_segment(cluster_id=i, size=s, segment_id=f"seg_{i}")
            for i, s in enumerate(sizes)
        ]

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.list_segments.return_value = segments

        service = _build_analytics_service(segment_repo=segment_repo)
        distribution = service.get_segment_distribution()

        assert len(distribution) == len(sizes), (
            f"Expected {len(sizes)} entries, got {len(distribution)}"
        )

        total_pct = sum(d["percentage"] for d in distribution)
        assert abs(total_pct - 100.0) < 1.0, (
            f"Percentages sum to {total_pct}, expected ~100.0"
        )

        for d in distribution:
            assert d["percentage"] >= 0.0, "Percentage must be non-negative"
            assert d["customer_count"] >= 0, "Customer count must be non-negative"

    # -- Property 20: Campaign metrics calculation ------------------------
    @settings(max_examples=50)
    @given(
        impressions=st.integers(min_value=1, max_value=100000),
        click_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        conversion_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    def test_campaign_metrics_calculation(self, impressions, click_ratio, conversion_ratio):
        """Property 20 -- CTR = clicks/impressions, conversion = conversions/clicks.

        Requirements: Campaign performance metrics are correctly calculated
        with CTR = clicks / impressions and conversion_rate = conversions / clicks.
        """
        clicks = int(impressions * click_ratio)
        conversions = int(clicks * conversion_ratio)

        assume(clicks <= impressions)
        assume(conversions <= clicks)

        if impressions > 0:
            expected_ctr = clicks / impressions
        else:
            expected_ctr = 0.0

        if clicks > 0:
            expected_conv_rate = conversions / clicks
        else:
            expected_conv_rate = 0.0

        now = datetime.utcnow()
        metrics = AdPerformanceMetrics(
            ad_id="ad_1",
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            click_through_rate=expected_ctr,
            conversion_rate=expected_conv_rate,
            segment_id="seg_0",
            measurement_period_start=now - timedelta(days=7),
            measurement_period_end=now,
        )

        ad = AdContent(
            ad_id="ad_1",
            segment_id="seg_0",
            campaign_id="camp_1",
            format=AdFormat.SHORT,
            content="Save today!",
            variation_number=1,
            use_case="cashback",
        )

        campaign = Campaign(
            campaign_id="camp_1",
            name="Test Campaign",
            target_segment_ids=["seg_0"],
            ad_content_ids=["ad_1"],
            status=CampaignStatus.ACTIVE,
            estimated_reach=1000,
        )

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        campaign_repo.get_campaign.return_value = campaign
        campaign_repo.get_campaign_ads.return_value = [ad]
        campaign_repo.get_performance_metrics.return_value = metrics

        service = _build_analytics_service(campaign_repo=campaign_repo)
        result = service.get_campaign_performance("camp_1")

        assert result["campaign_id"] == "camp_1"
        assert len(result["ads"]) == 1

        ad_metrics = result["ads"][0]
        assert ad_metrics["impressions"] == impressions
        assert ad_metrics["clicks"] == clicks
        assert ad_metrics["conversions"] == conversions

        assert abs(ad_metrics["click_through_rate"] - expected_ctr) < 1e-9, (
            f"CTR {ad_metrics['click_through_rate']} != expected {expected_ctr}"
        )
        if clicks > 0:
            assert abs(ad_metrics["conversion_rate"] - expected_conv_rate) < 1e-9, (
                f"Conversion rate {ad_metrics['conversion_rate']} != expected {expected_conv_rate}"
            )
        else:
            assert ad_metrics["conversion_rate"] == 0.0
