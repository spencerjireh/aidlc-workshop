"""Tests for TargetingEngine."""

import pytest
from unittest.mock import MagicMock

from src.models.campaign import Campaign, CampaignStatus
from src.models.service_models import CampaignActivationResult, ReachEstimate
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.targeting_engine import TargetingEngine


def _build_engine(segment_repo=None, campaign_repo=None):
    return TargetingEngine(
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


class TestCreateCampaign:
    def test_success(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.segment_exists.return_value = True
        segment_repo.get_segment_size.return_value = 200

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        engine = _build_engine(segment_repo=segment_repo, campaign_repo=campaign_repo)

        campaign = engine.create_campaign(
            campaign_name="Summer Promo",
            target_segment_ids=["seg_1"],
            ad_content_ids=["ad_1"],
        )
        assert campaign.status == CampaignStatus.DRAFT
        assert campaign.estimated_reach == 200
        campaign_repo.create_campaign.assert_called_once()

    def test_rejects_small_segment(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.segment_exists.return_value = True
        segment_repo.get_segment_size.return_value = 50

        engine = _build_engine(segment_repo=segment_repo)
        with pytest.raises(ValueError, match="minimum 100"):
            engine.create_campaign("Promo", ["seg_1"], ["ad_1"])

    def test_rejects_nonexistent_segment(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.segment_exists.return_value = False

        engine = _build_engine(segment_repo=segment_repo)
        with pytest.raises(ValueError, match="does not exist"):
            engine.create_campaign("Promo", ["missing"], ["ad_1"])

    def test_rejects_empty_segments(self):
        engine = _build_engine()
        with pytest.raises(ValueError, match="At least one target segment"):
            engine.create_campaign("Promo", [], ["ad_1"])

    def test_rejects_empty_ads(self):
        engine = _build_engine()
        with pytest.raises(ValueError, match="At least one ad content"):
            engine.create_campaign("Promo", ["seg_1"], [])

    def test_multiple_segments_reach(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.segment_exists.return_value = True
        # Called once during validation per segment, then once per segment in calculate_reach
        segment_repo.get_segment_size.side_effect = [200, 300, 200, 300]

        engine = _build_engine(segment_repo=segment_repo)
        campaign = engine.create_campaign("Promo", ["seg_1", "seg_2"], ["ad_1"])
        assert campaign.estimated_reach == 500


class TestCalculateReach:
    def test_sums_segment_sizes(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment_size.side_effect = [200, 300]

        engine = _build_engine(segment_repo=segment_repo)
        estimate = engine.calculate_reach(["seg_1", "seg_2"])

        assert isinstance(estimate, ReachEstimate)
        assert estimate.total_customers == 500
        assert estimate.segment_breakdown == {"seg_1": 200, "seg_2": 300}

    def test_single_segment(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment_size.return_value = 150

        engine = _build_engine(segment_repo=segment_repo)
        estimate = engine.calculate_reach(["seg_1"])
        assert estimate.total_customers == 150


class TestActivateCampaign:
    def _make_draft_campaign(self):
        return Campaign(
            campaign_id="camp_1",
            name="Summer Promo",
            target_segment_ids=["seg_1"],
            ad_content_ids=["ad_1", "ad_2"],
            status=CampaignStatus.DRAFT,
            estimated_reach=200,
        )

    def test_activates_successfully(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment_size.return_value = 200

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        campaign_repo.get_campaign.return_value = self._make_draft_campaign()

        engine = _build_engine(segment_repo=segment_repo, campaign_repo=campaign_repo)
        result = engine.activate_campaign("camp_1")

        assert isinstance(result, CampaignActivationResult)
        assert result.status == CampaignStatus.ACTIVE
        assert result.campaign_id == "camp_1"
        assert result.segments_targeted == ["seg_1"]
        assert result.ads_associated == ["ad_1", "ad_2"]
        campaign_repo.update_campaign.assert_called_once()

    def test_activate_sets_start_date(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment_size.return_value = 200

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        campaign_repo.get_campaign.return_value = self._make_draft_campaign()

        engine = _build_engine(segment_repo=segment_repo, campaign_repo=campaign_repo)
        result = engine.activate_campaign("camp_1")
        assert result.activated_at is not None

    def test_activate_missing_campaign_raises(self):
        campaign_repo = MagicMock(spec=CampaignDataRepository)
        campaign_repo.get_campaign.return_value = None

        engine = _build_engine(campaign_repo=campaign_repo)
        with pytest.raises(ValueError, match="not found"):
            engine.activate_campaign("missing")

    def test_activate_no_ads_raises(self):
        # Create a valid campaign, then clear its ads to simulate the edge case
        campaign = Campaign(
            campaign_id="camp_1",
            name="Promo",
            target_segment_ids=["seg_1"],
            ad_content_ids=["placeholder"],
            status=CampaignStatus.DRAFT,
            estimated_reach=0,
        )
        campaign.ad_content_ids = []  # Clear after construction to bypass Pydantic

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        campaign_repo.get_campaign.return_value = campaign

        engine = _build_engine(campaign_repo=campaign_repo)
        with pytest.raises(ValueError, match="no ad content"):
            engine.activate_campaign("camp_1")

    def test_activate_revalidates_segment_size(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment_size.return_value = 50  # below minimum

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        campaign_repo.get_campaign.return_value = self._make_draft_campaign()

        engine = _build_engine(segment_repo=segment_repo, campaign_repo=campaign_repo)
        with pytest.raises(ValueError, match="minimum 100"):
            engine.activate_campaign("camp_1")
