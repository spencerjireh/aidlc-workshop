"""Segmentation service orchestrating PCA, K-Means, and LLM engines."""

import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from src.engines.kmeans_engine import KMeansEngine
from src.engines.llm_engine import LLMEngine
from src.engines.pca_engine import PCAEngine
from src.models.customer import CustomerProfile
from src.models.segment import (
    ContributingFactor,
    CustomerSegmentAssignment,
    Segment,
    SegmentProfile,
)
from src.models.ml import ClusteringResult, PCAResult
from src.models.service_models import DataFormat, IngestionResult
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "age",
    "transaction_frequency",
    "average_transaction_value",
    "total_spend",
    "account_age_days",
]


class SegmentationService:
    """Service that orchestrates customer segmentation using PCA + K-Means + LLM."""

    def __init__(
        self,
        pca_engine: PCAEngine,
        kmeans_engine: KMeansEngine,
        llm_engine: LLMEngine,
        customer_repo: CustomerDataRepository,
        segment_repo: SegmentDataRepository,
    ):
        self.pca_engine = pca_engine
        self.kmeans_engine = kmeans_engine
        self.llm_engine = llm_engine
        self.customer_repo = customer_repo
        self.segment_repo = segment_repo

        # Internal state from the last pipeline run
        self._last_pca_result: Optional[PCAResult] = None
        self._last_clustering_result: Optional[ClusteringResult] = None
        self._last_customers: Optional[List[CustomerProfile]] = None
        self._last_pca_data: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def ingest_customer_data(
        self,
        data: List[Dict[str, Any]],
        format: DataFormat = DataFormat.JSON,
    ) -> IngestionResult:
        """Parse, validate, and store customer records.

        Invalid records are logged and skipped. Duplicates are merged using the
        incoming (most-recent) data.
        """
        successful = 0
        failed = 0
        duplicates_merged = 0
        errors: List[Dict[str, Any]] = []

        for idx, record in enumerate(data):
            try:
                parsed = self._parse_record(record, format)
                customer = CustomerProfile(**parsed)

                if self.customer_repo.customer_exists(customer.customer_id):
                    self.customer_repo.update_customer(customer.customer_id, customer)
                    duplicates_merged += 1
                else:
                    self.customer_repo.create_customer(customer)
                    successful += 1
            except Exception as exc:
                failed += 1
                errors.append({"record_index": idx, "error": str(exc)})
                logger.warning("Ingestion error at record %d: %s", idx, exc)

        return IngestionResult(
            total_records=len(data),
            successful=successful,
            failed=failed,
            duplicates_merged=duplicates_merged,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Segmentation pipeline
    # ------------------------------------------------------------------

    def create_segments(
        self,
        customer_ids: Optional[List[str]] = None,
        num_clusters: Optional[int] = None,
    ) -> List[Segment]:
        """Run the full PCA + K-Means + LLM segmentation pipeline."""

        # 1. Fetch customers
        if customer_ids:
            customers = [
                c
                for cid in customer_ids
                if (c := self.customer_repo.get_customer(cid)) is not None
            ]
        else:
            customers = self.customer_repo.list_customers()

        if len(customers) < 3:
            raise ValueError("At least 3 customers are required for segmentation")

        # 2. Extract numeric features
        feature_matrix = np.array(
            [self._extract_features(c) for c in customers], dtype=float
        )

        # 3. PCA
        pca_result = self.pca_engine.fit_transform(
            feature_matrix, FEATURE_NAMES, variance_threshold=0.8
        )
        pca_data = np.array(pca_result.transformed_data)

        # 4. Determine k
        if num_clusters is None:
            optimal_k = self.kmeans_engine.determine_optimal_k(pca_data, k_range=(3, min(10, len(customers))))
        else:
            optimal_k = num_clusters

        # 5. K-Means
        clustering_result = self.kmeans_engine.fit_predict(pca_data, n_clusters=optimal_k)

        # 6. Group customers by cluster
        cluster_groups: Dict[int, List[CustomerProfile]] = defaultdict(list)
        for i, label in enumerate(clustering_result.cluster_labels):
            cluster_groups[label].append(customers[i])

        # 7. Build segments
        segments: List[Segment] = []
        for cluster_id in range(clustering_result.n_clusters):
            cluster_customers = cluster_groups.get(cluster_id, [])
            if not cluster_customers:
                continue

            stats = self.kmeans_engine.get_cluster_statistics(cluster_id, cluster_customers)
            profile = self.llm_engine.generate_segment_profile(stats, cluster_id)

            # PCA component contributions
            pca_contributions: Dict[int, float] = {}
            for comp_idx in range(pca_result.n_components):
                pca_contributions[comp_idx] = float(
                    pca_result.explained_variance_ratio[comp_idx]
                )

            segment = Segment(
                segment_id=profile.segment_id,
                name=profile.name,
                description=profile.description,
                characteristics={
                    "average_age": stats.average_age,
                    "location_distribution": stats.location_distribution,
                    "preferred_payment_methods": stats.preferred_payment_methods,
                },
                cluster_id=cluster_id,
                centroid=list(clustering_result.centroids[cluster_id]),
                size=stats.size,
                average_transaction_value=stats.average_transaction_value,
                transaction_frequency=stats.average_transaction_frequency,
                top_merchant_categories=[
                    cat for cat, _ in stats.top_merchant_categories
                ],
                differentiating_factors=profile.differentiating_factors,
                pca_component_contributions=pca_contributions,
                version=1,
            )
            self.segment_repo.create_segment(segment)
            segments.append(segment)

        # 8. Store internal state
        self._last_pca_result = pca_result
        self._last_clustering_result = clustering_result
        self._last_customers = customers
        self._last_pca_data = pca_data

        return segments

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_customers_to_segments(
        self,
        customer_ids: List[str],
        segments: List[Segment],
    ) -> List[CustomerSegmentAssignment]:
        """Assign each customer to exactly one segment based on cluster membership."""

        if self._last_clustering_result is None or self._last_pca_data is None:
            raise ValueError("create_segments must be called before assigning customers")

        # Build lookup: cluster_id -> segment
        cluster_to_segment = {s.cluster_id: s for s in segments}

        assignments: List[CustomerSegmentAssignment] = []
        customers = self._last_customers or []

        # Build customer_id -> index mapping
        customer_index = {c.customer_id: i for i, c in enumerate(customers)}

        for cid in customer_ids:
            idx = customer_index.get(cid)
            if idx is None:
                logger.warning("Customer %s not found in last pipeline run", cid)
                continue

            label = int(self._last_clustering_result.cluster_labels[idx])
            segment = cluster_to_segment.get(label)
            if segment is None:
                continue

            point = self._last_pca_data[idx]
            centroid = np.array(segment.centroid)

            confidence = self.kmeans_engine.calculate_confidence_score(point, label)
            distance = float(np.linalg.norm(point - centroid))

            assignment = CustomerSegmentAssignment(
                assignment_id=f"assign_{uuid.uuid4().hex[:8]}",
                customer_id=cid,
                segment_id=segment.segment_id,
                confidence_score=confidence,
                distance_to_centroid=distance,
            )
            self.segment_repo.assign_customer_to_segment(assignment)
            assignments.append(assignment)

        return assignments

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain_assignment(
        self,
        customer_id: str,
        segment_id: str,
    ) -> CustomerSegmentAssignment:
        """Generate an LLM explanation and contributing factors for an assignment."""

        customer = self.customer_repo.get_customer(customer_id)
        if customer is None:
            raise ValueError(f"Customer {customer_id} not found")

        segment = self.segment_repo.get_segment(segment_id)
        if segment is None:
            raise ValueError(f"Segment {segment_id} not found")

        # Top 3 feature importances from the first principal component
        top_features = self.pca_engine.get_feature_importance(0)[:3]

        explanation = self.llm_engine.explain_cluster_assignment(
            customer, segment.centroid, top_features
        )

        feature_values = {
            "age": str(customer.age),
            "transaction_frequency": str(customer.transaction_frequency),
            "average_transaction_value": f"${customer.average_transaction_value:.2f}",
            "total_spend": f"${customer.total_spend:.2f}",
            "account_age_days": str(customer.account_age_days),
        }

        contributing_factors = []
        for fname, loading in top_features:
            contributing_factors.append(
                ContributingFactor(
                    factor_name=fname,
                    importance=min(abs(loading), 1.0),
                    data_point=feature_values.get(fname, "N/A"),
                    pca_loading=loading,
                )
            )

        # Update existing assignment if present, otherwise build a new one
        existing = self.segment_repo.get_customer_assignment(customer_id)
        if existing is not None:
            existing.explanation = explanation
            existing.contributing_factors = contributing_factors
            return existing

        # Fallback: create a minimal assignment record
        assignment = CustomerSegmentAssignment(
            assignment_id=f"assign_{uuid.uuid4().hex[:8]}",
            customer_id=customer_id,
            segment_id=segment_id,
            confidence_score=0.5,
            distance_to_centroid=0.0,
            explanation=explanation,
            contributing_factors=contributing_factors,
        )
        return assignment

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    def refine_segments(self, num_clusters: int) -> List[Segment]:
        """Re-cluster with a new k value, re-assign all customers, track versions."""

        if self._last_customers is None:
            raise ValueError("create_segments must be called before refining")

        # Clear old assignments
        self.segment_repo.clear_all_assignments()

        # Delete old segments from repo
        for seg in self.segment_repo.list_segments():
            self.segment_repo.delete_segment(seg.segment_id)

        # Re-run pipeline
        customer_ids = [c.customer_id for c in self._last_customers]
        segments = self.create_segments(customer_ids=customer_ids, num_clusters=num_clusters)

        # Re-assign
        self.assign_customers_to_segments(customer_ids, segments)

        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_features(customer: CustomerProfile) -> List[float]:
        return [
            float(customer.age),
            float(customer.transaction_frequency),
            float(customer.average_transaction_value),
            float(customer.total_spend),
            float(customer.account_age_days),
        ]

    @staticmethod
    def _parse_record(record: Dict[str, Any], format: DataFormat) -> Dict[str, Any]:
        """Normalise a raw record dict (handles CSV pipe-delimited lists)."""
        parsed = dict(record)
        if format == DataFormat.CSV:
            for field in ("merchant_categories", "preferred_payment_methods"):
                val = parsed.get(field)
                if isinstance(val, str):
                    parsed[field] = [v.strip() for v in val.split("|") if v.strip()]
        return parsed
