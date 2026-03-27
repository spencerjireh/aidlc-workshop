"""Segmentation endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_segment_repo, get_segmentation_service
from src.api.schemas import SegmentCreateRequest, SegmentRefineRequest
from src.repositories.segment_repository import SegmentDataRepository
from src.services.segmentation_service import SegmentationService

router = APIRouter(prefix="/segments", tags=["segments"])


@router.post("/create")
def create_segments(
    request: SegmentCreateRequest,
    service: SegmentationService = Depends(get_segmentation_service),
):
    """Create segments from customer data using PCA + K-Means + LLM."""
    try:
        segments = service.create_segments(
            customer_ids=request.customer_ids,
            num_clusters=request.num_clusters,
        )
        return [s.model_dump() for s in segments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
def list_segments(
    repo: SegmentDataRepository = Depends(get_segment_repo),
):
    """List all segments."""
    segments = repo.list_segments()
    return [s.model_dump() for s in segments]


@router.get("/{segment_id}")
def get_segment(
    segment_id: str,
    repo: SegmentDataRepository = Depends(get_segment_repo),
):
    """Get segment details."""
    segment = repo.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    return segment.model_dump()


@router.post("/refine")
def refine_segments(
    request: SegmentRefineRequest,
    service: SegmentationService = Depends(get_segmentation_service),
):
    """Refine segments with a new k value."""
    try:
        segments = service.refine_segments(num_clusters=request.num_clusters)
        return [s.model_dump() for s in segments]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{segment_id}/customers")
def get_segment_customers(
    segment_id: str,
    repo: SegmentDataRepository = Depends(get_segment_repo),
):
    """Get customers in a segment."""
    if not repo.segment_exists(segment_id):
        raise HTTPException(status_code=404, detail="Segment not found")
    customer_ids = repo.get_segment_customers(segment_id)
    return {"segment_id": segment_id, "customer_ids": customer_ids, "count": len(customer_ids)}
