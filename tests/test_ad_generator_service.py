"""Tests for AdGeneratorService."""

import pytest
from unittest.mock import MagicMock

from src.engines.llm_engine import LLMEngine
from src.models.campaign import AdContent, AdFormat
from src.models.segment import Segment, SegmentProfile
from src.models.service_models import ContentValidationResult
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.ad_generator_service import AdGeneratorService


def _make_segment(segment_id="segment_0"):
    return Segment(
        segment_id=segment_id,
        name="Young Spenders",
        description="Young, frequent spenders",
        characteristics={"average_age": 25.0},
        cluster_id=0,
        centroid=[0.1, 0.2],
        size=200,
        average_transaction_value=500.0,
        transaction_frequency=30.0,
        top_merchant_categories=["Food", "Shopping"],
        differentiating_factors=["High frequency", "Young"],
        pca_component_contributions={0: 0.6},
    )


def _make_ad(variation: int, fmt: AdFormat = AdFormat.SHORT) -> AdContent:
    limits = {AdFormat.SHORT: 50, AdFormat.MEDIUM: 150, AdFormat.LONG: 300}
    return AdContent(
        ad_id=f"ad_segment_0_{fmt.value}_{variation}",
        segment_id="segment_0",
        campaign_id="",
        format=fmt,
        content="Save big today!" if fmt == AdFormat.SHORT else "Save big today with our e-wallet offers.",
        variation_number=variation,
        use_case="cashback",
    )


def _build_service(llm=None, segment_repo=None, campaign_repo=None):
    return AdGeneratorService(
        llm_engine=llm or MagicMock(spec=LLMEngine),
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


class TestCreateCampaignAds:
    def test_generates_variations(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.return_value = [
            _make_ad(1), _make_ad(2), _make_ad(3)
        ]

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        service = _build_service(llm=llm, segment_repo=segment_repo, campaign_repo=campaign_repo)

        ads = service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_1",
            campaign_brief="Cashback promo",
            formats=[AdFormat.SHORT],
        )
        assert len(ads) == 3

    def test_enforces_minimum_variations(self):
        service = _build_service()
        with pytest.raises(ValueError, match="at least 3"):
            service.create_campaign_ads(
                segment_id="segment_0",
                campaign_id="camp_1",
                campaign_brief="Promo",
                formats=[AdFormat.SHORT],
                num_variations=1,
            )

    def test_multiple_formats(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.side_effect = [
            [_make_ad(1, AdFormat.SHORT), _make_ad(2, AdFormat.SHORT), _make_ad(3, AdFormat.SHORT)],
            [_make_ad(1, AdFormat.MEDIUM), _make_ad(2, AdFormat.MEDIUM), _make_ad(3, AdFormat.MEDIUM)],
        ]

        service = _build_service(llm=llm, segment_repo=segment_repo)
        ads = service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_1",
            campaign_brief="Cashback promo",
            formats=[AdFormat.SHORT, AdFormat.MEDIUM],
        )
        assert len(ads) == 6

    def test_campaign_id_overridden(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.return_value = [_make_ad(1), _make_ad(2), _make_ad(3)]

        service = _build_service(llm=llm, segment_repo=segment_repo)
        ads = service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_42",
            campaign_brief="Cashback promo",
            formats=[AdFormat.SHORT],
        )
        for ad in ads:
            assert ad.campaign_id == "camp_42"

    def test_ads_stored_in_repo(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.return_value = [_make_ad(1), _make_ad(2), _make_ad(3)]

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        service = _build_service(llm=llm, segment_repo=segment_repo, campaign_repo=campaign_repo)

        service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_1",
            campaign_brief="Promo",
            formats=[AdFormat.SHORT],
        )
        assert campaign_repo.create_ad_content.call_count == 3

    def test_invalid_segment_raises(self):
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = None
        service = _build_service(segment_repo=segment_repo)

        with pytest.raises(ValueError, match="not found"):
            service.create_campaign_ads(
                segment_id="missing",
                campaign_id="camp_1",
                campaign_brief="Promo",
                formats=[AdFormat.SHORT],
            )


class TestValidateAdContent:
    def test_approves_clean_content(self):
        service = _build_service()
        result = service.validate_ad_content("Save 20% on groceries this weekend!")
        assert result.is_approved is True
        assert len(result.issues) == 0

    def test_rejects_inappropriate_content(self):
        service = _build_service()
        result = service.validate_ad_content("Get rich with guaranteed returns!")
        assert result.is_approved is False
        assert len(result.issues) > 0

    def test_rejects_empty_content(self):
        service = _build_service()
        result = service.validate_ad_content("")
        assert result.is_approved is False

    def test_rejects_whitespace_only(self):
        service = _build_service()
        result = service.validate_ad_content("   ")
        assert result.is_approved is False
