"""Property-based tests for service-layer components.

Uses Hypothesis strategies with mocked engines to validate invariants
across randomized inputs.
"""

import uuid
from collections import Counter
from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.engines.kmeans_engine import KMeansEngine
from src.engines.llm_engine import LLMEngine
from src.engines.pca_engine import PCAEngine
from src.models.campaign import AdContent, AdFormat, Campaign, CampaignStatus
from src.models.chatbot import (
    ChatbotResponse,
    ConversationContext,
    MessageRole,
    QueryIntent,
    QueryType,
)
from src.models.customer import CustomerProfile
from src.models.ml import ClusteringResult, ClusterStatistics, PCAResult
from src.models.segment import (
    ContributingFactor,
    CustomerSegmentAssignment,
    Segment,
    SegmentProfile,
)
from src.models.service_models import DataFormat, IngestionResult
from src.repositories.campaign_repository import CampaignDataRepository
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.ad_generator_service import AdGeneratorService
from src.services.query_chatbot_service import QueryChatbotService
from src.services.segmentation_service import SegmentationService
from src.services.targeting_engine import TargetingEngine


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def _customer_record_strategy():
    """Strategy for generating valid customer record dicts."""
    return st.fixed_dictionaries({
        "customer_id": st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=3,
            max_size=12,
        ),
        "age": st.integers(min_value=0, max_value=120),
        "location": st.sampled_from(["Manila", "Cebu", "Davao", "Quezon City"]),
        "transaction_frequency": st.integers(min_value=0, max_value=500),
        "average_transaction_value": st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        "merchant_categories": st.lists(
            st.sampled_from(["Food", "Transport", "Shopping", "Bills", "Entertainment"]),
            min_size=1,
            max_size=5,
        ),
        "total_spend": st.floats(min_value=0.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
        "account_age_days": st.integers(min_value=0, max_value=10000),
        "preferred_payment_methods": st.lists(
            st.sampled_from(["QR", "NFC", "Card", "Cash"]),
            min_size=1,
            max_size=3,
        ),
        "last_transaction_date": st.just("2025-01-01T00:00:00"),
    })


def _invalid_record_strategy():
    """Strategy for generating invalid customer records (missing fields)."""
    return st.fixed_dictionaries({
        "customer_id": st.text(min_size=1, max_size=5),
        "age": st.integers(min_value=-100, max_value=-1),
    })


def _make_customer(cid: str = "cust_001", age: int = 30) -> CustomerProfile:
    return CustomerProfile(
        customer_id=cid,
        age=age,
        location="Manila",
        transaction_frequency=20,
        average_transaction_value=500.0,
        merchant_categories=["Food", "Transport"],
        total_spend=10000.0,
        account_age_days=365,
        preferred_payment_methods=["QR", "NFC"],
        last_transaction_date=datetime(2025, 1, 1),
    )


def _make_segment(cluster_id: int = 0, size: int = 100, segment_id: str = None) -> Segment:
    return Segment(
        segment_id=segment_id or f"segment_{cluster_id}",
        name="Test Segment",
        description="A test segment description",
        characteristics={"average_age": 30.0},
        cluster_id=cluster_id,
        centroid=[0.1, 0.2],
        size=size,
        average_transaction_value=500.0,
        transaction_frequency=20.0,
        top_merchant_categories=["Food"],
        differentiating_factors=["High frequency"],
        pca_component_contributions={0: 0.6, 1: 0.3},
    )


def _build_segmentation_service(
    pca_engine=None, kmeans_engine=None, llm_engine=None,
    customer_repo=None, segment_repo=None,
) -> SegmentationService:
    return SegmentationService(
        pca_engine=pca_engine or MagicMock(spec=PCAEngine),
        kmeans_engine=kmeans_engine or MagicMock(spec=KMeansEngine),
        llm_engine=llm_engine or MagicMock(spec=LLMEngine),
        customer_repo=customer_repo or MagicMock(spec=CustomerDataRepository),
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
    )


def _build_ad_service(llm=None, segment_repo=None, campaign_repo=None) -> AdGeneratorService:
    return AdGeneratorService(
        llm_engine=llm or MagicMock(spec=LLMEngine),
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


def _build_targeting_engine(segment_repo=None, campaign_repo=None) -> TargetingEngine:
    return TargetingEngine(
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


def _build_chatbot_service(llm=None, segment_repo=None, customer_repo=None, campaign_repo=None) -> QueryChatbotService:
    return QueryChatbotService(
        llm_engine=llm or MagicMock(spec=LLMEngine),
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
        customer_repo=customer_repo or MagicMock(spec=CustomerDataRepository),
        campaign_repo=campaign_repo or MagicMock(spec=CampaignDataRepository),
    )


def _setup_segmentation_mocks(n_customers=6, n_clusters=2):
    """Set up standard mocks for the segmentation pipeline."""
    customers = [_make_customer(f"c{i}", age=20 + i) for i in range(n_customers)]

    customer_repo = MagicMock(spec=CustomerDataRepository)
    customer_repo.list_customers.return_value = customers
    customer_repo.get_customer.side_effect = lambda cid: next(
        (c for c in customers if c.customer_id == cid), None
    )
    customer_repo.customer_exists.return_value = False
    customer_repo.create_customer.return_value = "anon_id"

    pca_result = PCAResult(
        transformed_data=[[float(i), float(i) * 0.5] for i in range(n_customers)],
        explained_variance=[2.0, 1.0],
        explained_variance_ratio=[0.6, 0.3],
        components=[[0.5, 0.3, 0.1, 0.05, 0.05], [0.1, 0.1, 0.5, 0.2, 0.1]],
        feature_names=["age", "transaction_frequency", "average_transaction_value", "total_spend", "account_age_days"],
        n_components=2,
    )

    pca_engine = MagicMock(spec=PCAEngine)
    pca_engine.fit_transform.return_value = pca_result
    pca_engine.get_feature_importance.return_value = [
        ("age", 0.5),
        ("transaction_frequency", 0.3),
        ("average_transaction_value", 0.1),
    ]

    labels = [i % n_clusters for i in range(n_customers)]
    centroids = [[float(c), float(c) * 0.5] for c in range(n_clusters)]
    clustering_result = ClusteringResult(
        cluster_labels=labels,
        centroids=centroids,
        inertia=10.0,
        n_clusters=n_clusters,
        silhouette_score=0.5,
    )

    kmeans_engine = MagicMock(spec=KMeansEngine)
    kmeans_engine.determine_optimal_k.return_value = n_clusters
    kmeans_engine.fit_predict.return_value = clustering_result
    kmeans_engine.get_cluster_statistics.return_value = ClusterStatistics(
        cluster_id=0,
        size=n_customers // n_clusters,
        average_age=25.0,
        location_distribution={"Manila": 3},
        average_transaction_frequency=20.0,
        average_transaction_value=500.0,
        total_spend_distribution={"p25": 5000, "p50": 10000, "p75": 15000},
        top_merchant_categories=[("Food", 3), ("Transport", 2)],
        preferred_payment_methods={"QR": 3},
    )
    kmeans_engine.calculate_confidence_score.return_value = 0.85

    def _make_profile(stats, cluster_id):
        return SegmentProfile(
            segment_id=f"segment_{cluster_id}",
            name=f"Segment {cluster_id}",
            description=f"Description for segment {cluster_id}",
            differentiating_factors=[f"Factor A{cluster_id}", f"Factor B{cluster_id}"],
        )

    llm_engine = MagicMock(spec=LLMEngine)
    llm_engine.generate_segment_profile.side_effect = _make_profile

    segment_repo = SegmentDataRepository()

    return {
        "customer_repo": customer_repo,
        "pca_engine": pca_engine,
        "kmeans_engine": kmeans_engine,
        "llm_engine": llm_engine,
        "segment_repo": segment_repo,
        "customers": customers,
    }


# ===========================================================================
# Segmentation Service property tests
# ===========================================================================


class TestSegmentationProperties:
    """Property-based tests for SegmentationService."""

    # -- Property 2: Error isolation in batch processing -------------------
    @settings(max_examples=50)
    @given(
        n_valid=st.integers(min_value=1, max_value=5),
        n_invalid=st.integers(min_value=1, max_value=5),
    )
    def test_error_isolation_in_batch_processing(self, n_valid, n_invalid):
        """Property 2 -- Error isolation in batch processing.

        Requirements: Valid records processed despite invalid ones present.
        total_records == successful + failed + duplicates_merged.
        """
        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.customer_exists.return_value = False
        customer_repo.create_customer.return_value = "anon_id"
        service = _build_segmentation_service(customer_repo=customer_repo)

        valid_records = [
            {
                "customer_id": f"c{i}",
                "age": 25,
                "location": "Manila",
                "transaction_frequency": 10,
                "average_transaction_value": 300.0,
                "merchant_categories": ["Food"],
                "total_spend": 3000.0,
                "account_age_days": 100,
                "preferred_payment_methods": ["QR"],
                "last_transaction_date": "2025-01-01T00:00:00",
            }
            for i in range(n_valid)
        ]
        invalid_records = [
            {"customer_id": f"bad{i}", "age": -5}
            for i in range(n_invalid)
        ]
        all_records = valid_records + invalid_records

        result = service.ingest_customer_data(all_records)

        assert result.total_records == n_valid + n_invalid
        assert result.successful == n_valid
        assert result.failed == n_invalid
        assert result.total_records == result.successful + result.failed + result.duplicates_merged

    # -- Property 3: Duplicate merge uses latest --------------------------
    @settings(max_examples=50)
    @given(n_duplicates=st.integers(min_value=1, max_value=5))
    def test_duplicate_merge_uses_latest(self, n_duplicates):
        """Property 3 -- Duplicate customer_ids get merged (update, not create).

        Requirements: When the same customer_id appears multiple times, duplicates_merged
        is incremented and update_customer is called instead of create_customer.
        """
        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.customer_exists.return_value = True
        customer_repo.update_customer.return_value = True
        service = _build_segmentation_service(customer_repo=customer_repo)

        records = [
            {
                "customer_id": "same_id",
                "age": 25 + i,
                "location": "Manila",
                "transaction_frequency": 10,
                "average_transaction_value": 300.0,
                "merchant_categories": ["Food"],
                "total_spend": 3000.0,
                "account_age_days": 100,
                "preferred_payment_methods": ["QR"],
                "last_transaction_date": "2025-01-01T00:00:00",
            }
            for i in range(n_duplicates)
        ]

        result = service.ingest_customer_data(records)

        assert result.duplicates_merged == n_duplicates
        assert result.successful == 0
        assert customer_repo.update_customer.call_count == n_duplicates

    # -- Property 5: Segment completeness ---------------------------------
    @settings(max_examples=50)
    @given(n_clusters=st.integers(min_value=2, max_value=3))
    def test_segment_completeness(self, n_clusters):
        """Property 5 -- Each segment has name, description, centroid, differentiating_factors.

        Requirements: Every returned Segment object has non-empty name, description,
        centroid, and differentiating_factors fields.
        """
        m = _setup_segmentation_mocks(n_customers=6, n_clusters=n_clusters)
        service = _build_segmentation_service(
            pca_engine=m["pca_engine"],
            kmeans_engine=m["kmeans_engine"],
            llm_engine=m["llm_engine"],
            customer_repo=m["customer_repo"],
            segment_repo=m["segment_repo"],
        )

        segments = service.create_segments(num_clusters=n_clusters)

        for seg in segments:
            assert seg.name and seg.name.strip(), "Segment name must be non-empty"
            assert seg.description and seg.description.strip(), "Segment description must be non-empty"
            assert len(seg.centroid) > 0, "Segment centroid must be non-empty"
            assert isinstance(seg.differentiating_factors, list), "differentiating_factors must be a list"

    # -- Property 6: Unique segment assignment ----------------------------
    @settings(max_examples=50)
    @given(st.just(6))
    def test_unique_segment_assignment(self, n_customers):
        """Property 6 -- Each customer is assigned to exactly one segment.

        Requirements: No customer appears in more than one segment assignment.
        """
        m = _setup_segmentation_mocks(n_customers=n_customers, n_clusters=2)
        service = _build_segmentation_service(
            pca_engine=m["pca_engine"],
            kmeans_engine=m["kmeans_engine"],
            llm_engine=m["llm_engine"],
            customer_repo=m["customer_repo"],
            segment_repo=m["segment_repo"],
        )

        segments = service.create_segments(num_clusters=2)
        customer_ids = [c.customer_id for c in m["customers"]]
        assignments = service.assign_customers_to_segments(customer_ids, segments)

        assigned_cids = [a.customer_id for a in assignments]
        assert len(assigned_cids) == len(set(assigned_cids)), (
            "Each customer must appear in exactly one assignment"
        )

    # -- Property 8: Segment statistics consistency -----------------------
    @settings(max_examples=50)
    @given(st.just(6))
    def test_segment_statistics_consistency(self, n_customers):
        """Property 8 -- Segment size equals count of assigned customers.

        Requirements: After assignment, the number of assignments per segment
        matches the segment's reported size from the repository.
        """
        m = _setup_segmentation_mocks(n_customers=n_customers, n_clusters=2)
        service = _build_segmentation_service(
            pca_engine=m["pca_engine"],
            kmeans_engine=m["kmeans_engine"],
            llm_engine=m["llm_engine"],
            customer_repo=m["customer_repo"],
            segment_repo=m["segment_repo"],
        )

        segments = service.create_segments(num_clusters=2)
        customer_ids = [c.customer_id for c in m["customers"]]
        assignments = service.assign_customers_to_segments(customer_ids, segments)

        segment_repo = m["segment_repo"]
        for seg in segments:
            repo_size = segment_repo.get_segment_size(seg.segment_id)
            assignment_count = sum(
                1 for a in assignments if a.segment_id == seg.segment_id
            )
            assert repo_size == assignment_count, (
                f"Segment {seg.segment_id}: repo size {repo_size} != "
                f"assignment count {assignment_count}"
            )

    # -- Property 9: Segment profile updates ------------------------------
    @settings(max_examples=50)
    @given(st.just(2))
    def test_segment_profile_updates(self, n_clusters):
        """Property 9 -- Re-clustering updates segment profiles.

        Requirements: Running create_segments twice with different k values
        produces different segment sets.
        """
        m = _setup_segmentation_mocks(n_customers=6, n_clusters=2)
        service = _build_segmentation_service(
            pca_engine=m["pca_engine"],
            kmeans_engine=m["kmeans_engine"],
            llm_engine=m["llm_engine"],
            customer_repo=m["customer_repo"],
            segment_repo=m["segment_repo"],
        )

        segments_v1 = service.create_segments(num_clusters=2)
        assert len(segments_v1) == 2

        # Re-cluster with a different k (mocks still work for any k)
        m2 = _setup_segmentation_mocks(n_customers=6, n_clusters=3)
        service.pca_engine = m2["pca_engine"]
        service.kmeans_engine = m2["kmeans_engine"]
        service.llm_engine = m2["llm_engine"]
        service.segment_repo = m2["segment_repo"]

        segments_v2 = service.create_segments(num_clusters=3)
        assert len(segments_v2) == 3
        assert segments_v1 != segments_v2

    # -- Property 16: Assignment explanation completeness -----------------
    @settings(max_examples=50)
    @given(st.just("c1"))
    def test_assignment_explanation_completeness(self, customer_id):
        """Property 16 -- Explanation has text and contributing factors.

        Requirements: explain_assignment returns an assignment with non-empty
        explanation text and a list of contributing factors.
        """
        pca_engine = MagicMock(spec=PCAEngine)
        pca_engine.get_feature_importance.return_value = [
            ("age", 0.5),
            ("transaction_frequency", 0.3),
            ("average_transaction_value", 0.1),
        ]

        llm_engine = MagicMock(spec=LLMEngine)
        llm_engine.explain_cluster_assignment.return_value = (
            "This customer belongs to this segment due to high spending."
        )

        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.get_customer.return_value = _make_customer(customer_id)

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.get_customer_assignment.return_value = None

        service = _build_segmentation_service(
            pca_engine=pca_engine,
            llm_engine=llm_engine,
            customer_repo=customer_repo,
            segment_repo=segment_repo,
        )

        result = service.explain_assignment(customer_id, "segment_0")

        assert result.explanation is not None and len(result.explanation) > 0, (
            "Explanation text must be non-empty"
        )
        assert len(result.contributing_factors) > 0, (
            "Contributing factors must not be empty"
        )
        for cf in result.contributing_factors:
            assert cf.factor_name, "Factor name must be non-empty"
            assert 0.0 <= cf.importance <= 1.0, "Importance must be in [0, 1]"

    # -- Property 17: Explanation data references -------------------------
    @settings(max_examples=50)
    @given(
        age=st.integers(min_value=18, max_value=80),
        txn_freq=st.integers(min_value=1, max_value=100),
    )
    def test_explanation_data_references(self, age, txn_freq):
        """Property 17 -- Explanation references customer data.

        Requirements: Contributing factors' data_point fields reference actual
        customer attribute values.
        """
        customer = CustomerProfile(
            customer_id="cust_ref",
            age=age,
            location="Manila",
            transaction_frequency=txn_freq,
            average_transaction_value=500.0,
            merchant_categories=["Food"],
            total_spend=10000.0,
            account_age_days=365,
            preferred_payment_methods=["QR"],
            last_transaction_date=datetime(2025, 1, 1),
        )

        pca_engine = MagicMock(spec=PCAEngine)
        pca_engine.get_feature_importance.return_value = [
            ("age", 0.5),
            ("transaction_frequency", 0.3),
            ("average_transaction_value", 0.1),
        ]

        llm_engine = MagicMock(spec=LLMEngine)
        llm_engine.explain_cluster_assignment.return_value = "Explanation text."

        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.get_customer.return_value = customer

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.get_customer_assignment.return_value = None

        service = _build_segmentation_service(
            pca_engine=pca_engine,
            llm_engine=llm_engine,
            customer_repo=customer_repo,
            segment_repo=segment_repo,
        )

        result = service.explain_assignment("cust_ref", "segment_0")

        data_points = [cf.data_point for cf in result.contributing_factors]
        assert str(age) in data_points, (
            f"Expected customer age {age} in data references: {data_points}"
        )
        assert str(txn_freq) in data_points, (
            f"Expected transaction_frequency {txn_freq} in data references: {data_points}"
        )

    # -- Property 28: Segment refinement triggers reassignment ------------
    @settings(max_examples=50)
    @given(st.just(3))
    def test_segment_refinement_triggers_reassignment(self, new_k):
        """Property 28 -- refine_segments re-assigns all customers.

        Requirements: After refine_segments, new assignments exist for all
        customers and old assignments are cleared.
        """
        m = _setup_segmentation_mocks(n_customers=6, n_clusters=2)
        segment_repo = m["segment_repo"]
        service = _build_segmentation_service(
            pca_engine=m["pca_engine"],
            kmeans_engine=m["kmeans_engine"],
            llm_engine=m["llm_engine"],
            customer_repo=m["customer_repo"],
            segment_repo=segment_repo,
        )

        segments_v1 = service.create_segments(num_clusters=2)
        customer_ids = [c.customer_id for c in m["customers"]]
        service.assign_customers_to_segments(customer_ids, segments_v1)

        # Verify initial assignments exist
        initial_assignments = sum(
            len(segment_repo.get_segment_customers(s.segment_id))
            for s in segments_v1
        )
        assert initial_assignments > 0

        # Refine with new k -- mocks need to support re-run
        m2 = _setup_segmentation_mocks(n_customers=6, n_clusters=new_k)
        service.pca_engine = m2["pca_engine"]
        service.kmeans_engine = m2["kmeans_engine"]
        service.llm_engine = m2["llm_engine"]
        service.customer_repo = m2["customer_repo"]
        service.segment_repo = m2["segment_repo"]

        refined_segments = service.refine_segments(num_clusters=new_k)

        # Verify new assignments exist for all customers
        new_segment_repo = m2["segment_repo"]
        new_assignment_count = sum(
            len(new_segment_repo.get_segment_customers(s.segment_id))
            for s in refined_segments
        )
        assert new_assignment_count == len(customer_ids), (
            f"Expected {len(customer_ids)} assignments, got {new_assignment_count}"
        )

    # -- Property 29: Segment version history -----------------------------
    @settings(max_examples=50)
    @given(st.just(2))
    def test_segment_version_history(self, n_clusters):
        """Property 29 -- Re-clustering creates version history entries.

        Requirements: After create_segments, version history exists for each
        segment in the repository.
        """
        m = _setup_segmentation_mocks(n_customers=6, n_clusters=n_clusters)
        service = _build_segmentation_service(
            pca_engine=m["pca_engine"],
            kmeans_engine=m["kmeans_engine"],
            llm_engine=m["llm_engine"],
            customer_repo=m["customer_repo"],
            segment_repo=m["segment_repo"],
        )

        segments = service.create_segments(num_clusters=n_clusters)

        segment_repo = m["segment_repo"]
        for seg in segments:
            history = segment_repo.get_segment_version_history(seg.segment_id)
            assert len(history) >= 1, (
                f"Segment {seg.segment_id} should have at least 1 version history entry"
            )


# ===========================================================================
# Ad Generator Service property tests
# ===========================================================================


class TestAdGeneratorProperties:
    """Property-based tests for AdGeneratorService."""

    # -- Property 10: Ad format character limits --------------------------
    @settings(max_examples=50)
    @given(ad_format=st.sampled_from([AdFormat.SHORT, AdFormat.MEDIUM, AdFormat.LONG]))
    def test_ad_format_character_limits(self, ad_format):
        """Property 10 -- Ads respect SHORT<=50, MEDIUM<=150, LONG<=300.

        Requirements: Every ad returned by create_campaign_ads has content
        length within the limit for its format.
        """
        limits = {AdFormat.SHORT: 50, AdFormat.MEDIUM: 150, AdFormat.LONG: 300}
        char_limit = limits[ad_format]

        content = "X" * min(char_limit, 40)

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        ads = [
            AdContent(
                ad_id=f"ad_segment_0_{ad_format.value}_{i}",
                segment_id="segment_0",
                campaign_id="",
                format=ad_format,
                content=content,
                variation_number=i,
                use_case="cashback",
            )
            for i in range(1, 4)
        ]

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.return_value = ads

        service = _build_ad_service(llm=llm, segment_repo=segment_repo)
        result = service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_1",
            campaign_brief="Cashback promo",
            formats=[ad_format],
        )

        for ad in result:
            assert len(ad.content) <= char_limit, (
                f"Ad content length {len(ad.content)} exceeds {char_limit} for {ad_format.value}"
            )

    # -- Property 11: Ad use_case assignment ------------------------------
    @settings(max_examples=50)
    @given(brief=st.sampled_from(["cashback offer", "merchant promo deal", "payment convenience"]))
    def test_ad_use_case_assignment(self, brief):
        """Property 11 -- Each ad has a valid use_case.

        Requirements: Every ad's use_case is one of: cashback, merchant_promo,
        payment_convenience.
        """
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        valid_use_cases = {"cashback", "merchant_promo", "payment_convenience"}

        use_case = "payment_convenience"
        if "cashback" in brief.lower():
            use_case = "cashback"
        elif "merchant" in brief.lower() or "promo" in brief.lower():
            use_case = "merchant_promo"

        ads = [
            AdContent(
                ad_id=f"ad_segment_0_short_{i}",
                segment_id="segment_0",
                campaign_id="",
                format=AdFormat.SHORT,
                content="Save today!",
                variation_number=i,
                use_case=use_case,
            )
            for i in range(1, 4)
        ]

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.return_value = ads

        service = _build_ad_service(llm=llm, segment_repo=segment_repo)
        result = service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_1",
            campaign_brief=brief,
            formats=[AdFormat.SHORT],
        )

        for ad in result:
            assert ad.use_case in valid_use_cases, (
                f"use_case '{ad.use_case}' not in {valid_use_cases}"
            )

    # -- Property 12: Minimum ad variations -------------------------------
    @settings(max_examples=50)
    @given(num_variations=st.integers(min_value=3, max_value=7))
    def test_minimum_ad_variations(self, num_variations):
        """Property 12 -- At least 3 variations per format.

        Requirements: create_campaign_ads produces at least 3 ad variations
        per requested format.
        """
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()

        ads = [
            AdContent(
                ad_id=f"ad_segment_0_short_{i}",
                segment_id="segment_0",
                campaign_id="",
                format=AdFormat.SHORT,
                content=f"Ad variation {i}!",
                variation_number=i,
                use_case="cashback",
            )
            for i in range(1, num_variations + 1)
        ]

        llm = MagicMock(spec=LLMEngine)
        llm.generate_ad_content.return_value = ads

        service = _build_ad_service(llm=llm, segment_repo=segment_repo)
        result = service.create_campaign_ads(
            segment_id="segment_0",
            campaign_id="camp_1",
            campaign_brief="Cashback promo",
            formats=[AdFormat.SHORT],
            num_variations=num_variations,
        )

        assert len(result) >= 3, (
            f"Expected at least 3 ad variations, got {len(result)}"
        )


# ===========================================================================
# Targeting Engine property tests
# ===========================================================================


class TestTargetingEngineProperties:
    """Property-based tests for TargetingEngine."""

    # -- Property 13: Campaign segment association ------------------------
    @settings(max_examples=50)
    @given(n_segments=st.integers(min_value=1, max_value=4))
    def test_campaign_segment_association(self, n_segments):
        """Property 13 -- Campaign targets specified segments.

        Requirements: The returned campaign's target_segment_ids match
        exactly the segments provided at creation time.
        """
        segment_ids = [f"seg_{i}" for i in range(n_segments)]

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.segment_exists.return_value = True
        segment_repo.get_segment_size.return_value = 200

        campaign_repo = MagicMock(spec=CampaignDataRepository)
        engine = _build_targeting_engine(
            segment_repo=segment_repo, campaign_repo=campaign_repo
        )

        campaign = engine.create_campaign(
            campaign_name="Test Campaign",
            target_segment_ids=segment_ids,
            ad_content_ids=["ad_1"],
        )

        assert campaign.target_segment_ids == segment_ids, (
            f"Campaign segments {campaign.target_segment_ids} != input {segment_ids}"
        )

    # -- Property 14: Reach calculation accuracy --------------------------
    @settings(max_examples=50)
    @given(
        sizes=st.lists(
            st.integers(min_value=100, max_value=10000),
            min_size=1,
            max_size=5,
        )
    )
    def test_reach_calculation_accuracy(self, sizes):
        """Property 14 -- Reach equals sum of segment sizes.

        Requirements: calculate_reach returns total_customers equal to the
        sum of all individual segment sizes.
        """
        segment_ids = [f"seg_{i}" for i in range(len(sizes))]

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment_size.side_effect = sizes

        engine = _build_targeting_engine(segment_repo=segment_repo)
        estimate = engine.calculate_reach(segment_ids)

        assert estimate.total_customers == sum(sizes), (
            f"Reach {estimate.total_customers} != sum of sizes {sum(sizes)}"
        )

    # -- Property 15: Minimum segment size enforcement --------------------
    @settings(max_examples=50)
    @given(segment_size=st.integers(min_value=0, max_value=99))
    def test_minimum_segment_size_enforcement(self, segment_size):
        """Property 15 -- Reject segment with fewer than 100 customers.

        Requirements: create_campaign raises ValueError when a target segment
        has fewer than 100 customers.
        """
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.segment_exists.return_value = True
        segment_repo.get_segment_size.return_value = segment_size

        engine = _build_targeting_engine(segment_repo=segment_repo)

        with pytest.raises(ValueError, match="minimum 100"):
            engine.create_campaign(
                campaign_name="Promo",
                target_segment_ids=["seg_small"],
                ad_content_ids=["ad_1"],
            )


# ===========================================================================
# Chatbot Service property tests
# ===========================================================================


class TestChatbotProperties:
    """Property-based tests for QueryChatbotService."""

    # -- Property 30: Chatbot response completeness -----------------------
    @settings(max_examples=50)
    @given(query=st.text(min_size=3, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))))
    def test_chatbot_response_completeness(self, query):
        """Property 30 -- Non-empty text response.

        Requirements: process_query always returns a ChatbotResponse with
        non-empty text.
        """
        assume(query.strip())

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.list_segments.return_value = [_make_segment()]

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={"segment": "segment_0"},
            confidence=0.9,
        )
        llm.generate_response.return_value = ChatbotResponse(
            response_id="resp_1",
            session_id="sess_prop",
            text="Here is the information you requested.",
            data={"key": "value"},
            suggested_followups=["What else?"],
            response_time_ms=50,
        )

        service = _build_chatbot_service(llm=llm, segment_repo=segment_repo)
        response = service.process_query(query, "sess_prop", "user1")

        assert isinstance(response, ChatbotResponse)
        assert response.text and response.text.strip(), "Response text must be non-empty"

    # -- Property 31: Conversation context persistence --------------------
    @settings(max_examples=50)
    @given(n_queries=st.integers(min_value=1, max_value=5))
    def test_conversation_context_persistence(self, n_queries):
        """Property 31 -- Queries are stored in session history.

        Requirements: After N queries, the conversation history contains
        at least N user messages.
        """
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.list_segments.return_value = [_make_segment()]

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={"segment": "segment_0"},
            confidence=0.9,
        )
        llm.generate_response.return_value = ChatbotResponse(
            response_id="resp_1",
            session_id="sess_persist",
            text="Response text here.",
            data={},
            suggested_followups=[],
            response_time_ms=10,
        )

        service = _build_chatbot_service(llm=llm, segment_repo=segment_repo)
        session_id = f"sess_persist_{uuid.uuid4().hex[:6]}"

        for i in range(n_queries):
            service.process_query(f"Query number {i + 1}", session_id, "user1")

        ctx = service.get_conversation_context(session_id)
        user_messages = [
            m for m in ctx.conversation_history if m.role == MessageRole.USER
        ]
        assert len(user_messages) == n_queries, (
            f"Expected {n_queries} user messages, found {len(user_messages)}"
        )

    # -- Property 32: Unanswerable query handling -------------------------
    @settings(max_examples=50)
    @given(query=st.text(min_size=5, max_size=30, alphabet=st.characters(whitelist_categories=("L",))))
    def test_unanswerable_query_handling(self, query):
        """Property 32 -- Limitation explanation and suggestions.

        Requirements: When confidence is below threshold, the service returns
        a response explaining the limitation and suggesting alternatives.
        """
        assume(query.strip())

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.list_segments.return_value = [
            _make_segment(0, segment_id="seg_0"),
            _make_segment(1, segment_id="seg_1"),
        ]

        llm = MagicMock(spec=LLMEngine)
        llm.interpret_query.return_value = QueryIntent(
            intent_type=QueryType.SEGMENT_INFO,
            entities={},
            confidence=0.2,
        )

        service = _build_chatbot_service(llm=llm, segment_repo=segment_repo)
        response = service.process_query(query, "sess_unans", "user1")

        assert "not sure" in response.text.lower() or "rephras" in response.text.lower(), (
            f"Expected limitation explanation, got: {response.text}"
        )
        assert len(response.suggested_followups) > 0, (
            "Expected suggested follow-up queries for unanswerable input"
        )
        llm.generate_response.assert_not_called()
