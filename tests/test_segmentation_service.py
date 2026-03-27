"""Tests for SegmentationService."""

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.engines.kmeans_engine import KMeansEngine
from src.engines.llm_engine import LLMEngine
from src.engines.pca_engine import PCAEngine
from src.models.customer import CustomerProfile
from src.models.ml import ClusteringResult, ClusterStatistics, PCAResult
from src.models.segment import (
    ContributingFactor,
    CustomerSegmentAssignment,
    Segment,
    SegmentProfile,
)
from src.models.service_models import DataFormat, IngestionResult
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.segmentation_service import SegmentationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_segment(cluster_id: int = 0, size: int = 100) -> Segment:
    return Segment(
        segment_id=f"segment_{cluster_id}",
        name="Test Segment",
        description="A test segment",
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


def _build_service(
    pca=None, kmeans=None, llm=None, customer_repo=None, segment_repo=None,
    pca_engine=None, kmeans_engine=None, llm_engine=None,
):
    return SegmentationService(
        pca_engine=pca_engine or pca or MagicMock(spec=PCAEngine),
        kmeans_engine=kmeans_engine or kmeans or MagicMock(spec=KMeansEngine),
        llm_engine=llm_engine or llm or MagicMock(spec=LLMEngine),
        customer_repo=customer_repo or MagicMock(spec=CustomerDataRepository),
        segment_repo=segment_repo or MagicMock(spec=SegmentDataRepository),
    )


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_ingest_valid_records(self):
        repo = MagicMock(spec=CustomerDataRepository)
        repo.customer_exists.return_value = False
        repo.create_customer.return_value = "anon_id"
        service = _build_service(customer_repo=repo)

        records = [
            {
                "customer_id": "c1",
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
        ]
        result = service.ingest_customer_data(records)
        assert isinstance(result, IngestionResult)
        assert result.successful == 1
        assert result.failed == 0

    def test_ingest_invalid_records_continue(self):
        repo = MagicMock(spec=CustomerDataRepository)
        repo.customer_exists.return_value = False
        repo.create_customer.return_value = "anon_id"
        service = _build_service(customer_repo=repo)

        records = [
            {"customer_id": "c1", "age": -5},  # invalid age
            {
                "customer_id": "c2",
                "age": 25,
                "location": "Manila",
                "transaction_frequency": 10,
                "average_transaction_value": 300.0,
                "merchant_categories": ["Food"],
                "total_spend": 3000.0,
                "account_age_days": 100,
                "preferred_payment_methods": ["QR"],
                "last_transaction_date": "2025-01-01T00:00:00",
            },
        ]
        result = service.ingest_customer_data(records)
        assert result.failed == 1
        assert result.successful == 1
        assert result.total_records == 2

    def test_ingest_duplicate_merge(self):
        repo = MagicMock(spec=CustomerDataRepository)
        repo.customer_exists.return_value = True
        repo.update_customer.return_value = True
        service = _build_service(customer_repo=repo)

        records = [
            {
                "customer_id": "c1",
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
        ]
        result = service.ingest_customer_data(records)
        assert result.duplicates_merged == 1
        assert result.successful == 0

    def test_ingest_csv_pipe_delimited(self):
        repo = MagicMock(spec=CustomerDataRepository)
        repo.customer_exists.return_value = False
        repo.create_customer.return_value = "anon_id"
        service = _build_service(customer_repo=repo)

        records = [
            {
                "customer_id": "c1",
                "age": 25,
                "location": "Manila",
                "transaction_frequency": 10,
                "average_transaction_value": 300.0,
                "merchant_categories": "Food|Transport|Shopping",
                "total_spend": 3000.0,
                "account_age_days": 100,
                "preferred_payment_methods": "QR|NFC",
                "last_transaction_date": "2025-01-01T00:00:00",
            }
        ]
        result = service.ingest_customer_data(records, format=DataFormat.CSV)
        assert result.successful == 1
        # Verify parsed lists were passed to CustomerProfile
        call_args = repo.create_customer.call_args[0][0]
        assert isinstance(call_args.merchant_categories, list)
        assert len(call_args.merchant_categories) == 3

    def test_ingest_empty_data(self):
        service = _build_service()
        result = service.ingest_customer_data([])
        assert result.total_records == 0
        assert result.successful == 0


# ---------------------------------------------------------------------------
# Segmentation pipeline tests
# ---------------------------------------------------------------------------

class TestCreateSegments:
    def _setup_mocks(self, n_customers=6, n_clusters=2):
        customers = [_make_customer(f"c{i}", age=20 + i) for i in range(n_customers)]

        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.list_customers.return_value = customers

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

        labels = [0] * (n_customers // 2) + [1] * (n_customers - n_customers // 2)
        clustering_result = ClusteringResult(
            cluster_labels=labels,
            centroids=[[1.0, 0.5], [4.0, 2.0]],
            inertia=10.0,
            n_clusters=n_clusters,
            silhouette_score=0.5,
        )

        kmeans_engine = MagicMock(spec=KMeansEngine)
        kmeans_engine.determine_optimal_k.return_value = n_clusters
        kmeans_engine.fit_predict.return_value = clustering_result
        kmeans_engine.get_cluster_statistics.return_value = ClusterStatistics(
            cluster_id=0,
            size=n_customers // 2,
            average_age=25.0,
            location_distribution={"Manila": 3},
            average_transaction_frequency=20.0,
            average_transaction_value=500.0,
            total_spend_distribution={"p25": 5000, "p50": 10000, "p75": 15000},
            top_merchant_categories=[("Food", 3), ("Transport", 2)],
            preferred_payment_methods={"QR": 3},
        )
        kmeans_engine.calculate_confidence_score.return_value = 0.85

        llm_engine = MagicMock(spec=LLMEngine)
        llm_engine.generate_segment_profile.return_value = SegmentProfile(
            segment_id="segment_0",
            name="Test Segment",
            description="A test segment description",
            differentiating_factors=["High frequency", "Young demographic"],
        )

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.create_segment.return_value = "segment_0"

        return {
            "customer_repo": customer_repo,
            "pca_engine": pca_engine,
            "kmeans_engine": kmeans_engine,
            "llm_engine": llm_engine,
            "segment_repo": segment_repo,
            "customers": customers,
        }

    def test_full_pipeline(self):
        m = self._setup_mocks()
        service = _build_service(**{k: v for k, v in m.items() if k != "customers"})

        segments = service.create_segments()
        assert len(segments) == 2
        m["pca_engine"].fit_transform.assert_called_once()
        m["kmeans_engine"].determine_optimal_k.assert_called_once()
        m["kmeans_engine"].fit_predict.assert_called_once()
        assert m["llm_engine"].generate_segment_profile.call_count == 2

    def test_explicit_k_skips_optimal(self):
        m = self._setup_mocks()
        service = _build_service(**{k: v for k, v in m.items() if k != "customers"})

        segments = service.create_segments(num_clusters=2)
        m["kmeans_engine"].determine_optimal_k.assert_not_called()
        m["kmeans_engine"].fit_predict.assert_called_once()

    def test_segments_stored_in_repo(self):
        m = self._setup_mocks()
        service = _build_service(**{k: v for k, v in m.items() if k != "customers"})

        service.create_segments()
        assert m["segment_repo"].create_segment.call_count == 2

    def test_too_few_customers_raises(self):
        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.list_customers.return_value = [_make_customer("c1")]
        service = _build_service(customer_repo=customer_repo)

        with pytest.raises(ValueError, match="At least 3"):
            service.create_segments()


# ---------------------------------------------------------------------------
# Assignment tests
# ---------------------------------------------------------------------------

class TestAssignment:
    def test_assign_customers(self):
        m = TestCreateSegments()._setup_mocks()
        service = _build_service(**{k: v for k, v in m.items() if k != "customers"})
        segments = service.create_segments()

        customer_ids = [c.customer_id for c in m["customers"]]
        assignments = service.assign_customers_to_segments(customer_ids, segments)

        assert len(assignments) == 6
        for a in assignments:
            assert 0.0 <= a.confidence_score <= 1.0
            assert a.distance_to_centroid >= 0.0

    def test_each_customer_one_segment(self):
        m = TestCreateSegments()._setup_mocks()
        service = _build_service(**{k: v for k, v in m.items() if k != "customers"})
        segments = service.create_segments()

        customer_ids = [c.customer_id for c in m["customers"]]
        assignments = service.assign_customers_to_segments(customer_ids, segments)

        assigned_ids = [a.customer_id for a in assignments]
        assert len(assigned_ids) == len(set(assigned_ids))

    def test_assign_before_create_raises(self):
        service = _build_service()
        with pytest.raises(ValueError, match="create_segments must be called"):
            service.assign_customers_to_segments(["c1"], [_make_segment()])


# ---------------------------------------------------------------------------
# Explain assignment tests
# ---------------------------------------------------------------------------

class TestExplainAssignment:
    def test_explain_uses_pca_and_llm(self):
        pca_engine = MagicMock(spec=PCAEngine)
        pca_engine.get_feature_importance.return_value = [
            ("age", 0.5),
            ("transaction_frequency", 0.3),
            ("average_transaction_value", 0.1),
        ]

        llm_engine = MagicMock(spec=LLMEngine)
        llm_engine.explain_cluster_assignment.return_value = "This customer is young and active."

        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.get_customer.return_value = _make_customer("c1")

        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = _make_segment()
        segment_repo.get_customer_assignment.return_value = None

        service = _build_service(
            pca=pca_engine,
            llm=llm_engine,
            customer_repo=customer_repo,
            segment_repo=segment_repo,
        )

        result = service.explain_assignment("c1", "segment_0")
        assert result.explanation == "This customer is young and active."
        assert len(result.contributing_factors) == 3
        pca_engine.get_feature_importance.assert_called_once_with(0)
        llm_engine.explain_cluster_assignment.assert_called_once()

    def test_explain_missing_customer_raises(self):
        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.get_customer.return_value = None
        service = _build_service(customer_repo=customer_repo)

        with pytest.raises(ValueError, match="not found"):
            service.explain_assignment("missing", "segment_0")

    def test_explain_missing_segment_raises(self):
        customer_repo = MagicMock(spec=CustomerDataRepository)
        customer_repo.get_customer.return_value = _make_customer("c1")
        segment_repo = MagicMock(spec=SegmentDataRepository)
        segment_repo.get_segment.return_value = None
        service = _build_service(customer_repo=customer_repo, segment_repo=segment_repo)

        with pytest.raises(ValueError, match="not found"):
            service.explain_assignment("c1", "missing")


# ---------------------------------------------------------------------------
# Refinement tests
# ---------------------------------------------------------------------------

class TestRefinement:
    def test_refine_clears_assignments_and_segments(self):
        m = TestCreateSegments()._setup_mocks()
        segment_repo = m["segment_repo"]
        segment_repo.list_segments.return_value = [_make_segment(0), _make_segment(1)]
        segment_repo.delete_segment.return_value = True

        service = _build_service(**{k: v for k, v in m.items() if k != "customers"})
        service.create_segments()

        # Now refine
        service.refine_segments(num_clusters=3)
        segment_repo.clear_all_assignments.assert_called_once()

    def test_refine_before_create_raises(self):
        service = _build_service()
        with pytest.raises(ValueError, match="create_segments must be called"):
            service.refine_segments(num_clusters=3)
