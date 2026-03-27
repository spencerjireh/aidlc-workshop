"""Unit tests for analytics service (Task 15.4)."""

import pytest
from datetime import datetime, timedelta

from src.services.analytics_service import AnalyticsService
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository
from src.models.segment import Segment
from src.models.campaign import (
    Campaign, AdContent, AdFormat, AdPerformanceMetrics, CampaignStatus,
)


def make_segment(segment_id, name, size):
    return Segment(
        segment_id=segment_id,
        name=name,
        description=f"Description for {name}",
        characteristics={"key": "value"},
        cluster_id=0,
        centroid=[1.0, 2.0],
        size=size,
        average_transaction_value=500.0,
        transaction_frequency=10.0,
        top_merchant_categories=["Food"],
        differentiating_factors=["Factor A"],
        pca_component_contributions={0: 0.5},
    )


class TestAnalyticsService:
    def setup_method(self):
        self.segment_repo = SegmentDataRepository()
        self.campaign_repo = CampaignDataRepository()
        self.service = AnalyticsService(
            segment_repo=self.segment_repo,
            campaign_repo=self.campaign_repo,
        )

    def test_segment_distribution_multiple_segments(self):
        self.segment_repo.create_segment(make_segment("s1", "Segment A", 300))
        self.segment_repo.create_segment(make_segment("s2", "Segment B", 200))
        self.segment_repo.create_segment(make_segment("s3", "Segment C", 500))

        dist = self.service.get_segment_distribution()
        assert len(dist) == 3
        assert any(d["segment_id"] == "s1" and d["customer_count"] == 300 for d in dist)

    def test_segment_distribution_percentages_sum_to_100(self):
        self.segment_repo.create_segment(make_segment("s1", "A", 300))
        self.segment_repo.create_segment(make_segment("s2", "B", 200))
        self.segment_repo.create_segment(make_segment("s3", "C", 500))

        dist = self.service.get_segment_distribution()
        total_pct = sum(d["percentage"] for d in dist)
        assert abs(total_pct - 100.0) < 0.1

    def test_segment_distribution_empty(self):
        dist = self.service.get_segment_distribution()
        assert dist == []

    def test_campaign_performance_found(self):
        campaign = Campaign(
            campaign_id="c1",
            name="Test Campaign",
            target_segment_ids=["s1"],
            ad_content_ids=["a1"],
            estimated_reach=100,
        )
        self.campaign_repo.create_campaign(campaign)

        ad = AdContent(
            ad_id="a1",
            segment_id="s1",
            campaign_id="c1",
            format=AdFormat.SHORT,
            content="Buy now!",
            variation_number=1,
            use_case="cashback",
        )
        self.campaign_repo.create_ad_content(ad)

        metrics = AdPerformanceMetrics(
            ad_id="a1",
            impressions=1000,
            clicks=100,
            conversions=10,
            click_through_rate=0.1,
            conversion_rate=0.1,
            segment_id="s1",
            measurement_period_start=datetime.utcnow() - timedelta(days=7),
            measurement_period_end=datetime.utcnow(),
        )
        self.campaign_repo.store_performance_metrics(metrics)

        perf = self.service.get_campaign_performance("c1")
        assert perf["campaign_id"] == "c1"
        assert len(perf["ads"]) == 1
        assert perf["ads"][0]["impressions"] == 1000
        assert perf["ads"][0]["click_through_rate"] == 0.1

    def test_campaign_performance_not_found(self):
        with pytest.raises(ValueError, match="Campaign not found"):
            self.service.get_campaign_performance("nonexistent")

    def test_campaign_performance_no_metrics(self):
        campaign = Campaign(
            campaign_id="c2",
            name="No Metrics",
            target_segment_ids=["s1"],
            ad_content_ids=["a2"],
            estimated_reach=50,
        )
        self.campaign_repo.create_campaign(campaign)

        ad = AdContent(
            ad_id="a2",
            segment_id="s1",
            campaign_id="c2",
            format=AdFormat.MEDIUM,
            content="Great deal!",
            variation_number=1,
            use_case="merchant_promo",
        )
        self.campaign_repo.create_ad_content(ad)

        perf = self.service.get_campaign_performance("c2")
        assert perf["ads"][0]["impressions"] == 0
        assert perf["ads"][0]["click_through_rate"] == 0.0

    def test_ctr_calculation(self):
        campaign = Campaign(
            campaign_id="c3",
            name="CTR Test",
            target_segment_ids=["s1"],
            ad_content_ids=["a3"],
            estimated_reach=200,
        )
        self.campaign_repo.create_campaign(campaign)

        ad = AdContent(
            ad_id="a3",
            segment_id="s1",
            campaign_id="c3",
            format=AdFormat.SHORT,
            content="Save big!",
            variation_number=1,
            use_case="cashback",
        )
        self.campaign_repo.create_ad_content(ad)

        metrics = AdPerformanceMetrics(
            ad_id="a3",
            impressions=500,
            clicks=50,
            conversions=5,
            click_through_rate=0.1,
            conversion_rate=0.1,
            segment_id="s1",
            measurement_period_start=datetime.utcnow() - timedelta(days=7),
            measurement_period_end=datetime.utcnow(),
        )
        self.campaign_repo.store_performance_metrics(metrics)

        perf = self.service.get_campaign_performance("c3")
        assert abs(perf["ads"][0]["click_through_rate"] - 0.1) < 0.01
        assert abs(perf["ads"][0]["conversion_rate"] - 0.1) < 0.01

    def test_zero_clicks_conversion_rate(self):
        campaign = Campaign(
            campaign_id="c4",
            name="Zero Clicks",
            target_segment_ids=["s1"],
            ad_content_ids=["a4"],
            estimated_reach=100,
        )
        self.campaign_repo.create_campaign(campaign)

        ad = AdContent(
            ad_id="a4",
            segment_id="s1",
            campaign_id="c4",
            format=AdFormat.SHORT,
            content="Free stuff!",
            variation_number=1,
            use_case="cashback",
        )
        self.campaign_repo.create_ad_content(ad)

        metrics = AdPerformanceMetrics(
            ad_id="a4",
            impressions=100,
            clicks=0,
            conversions=0,
            click_through_rate=0.0,
            conversion_rate=0.0,
            segment_id="s1",
            measurement_period_start=datetime.utcnow() - timedelta(days=7),
            measurement_period_end=datetime.utcnow(),
        )
        self.campaign_repo.store_performance_metrics(metrics)

        perf = self.service.get_campaign_performance("c4")
        assert perf["ads"][0]["conversion_rate"] == 0.0
