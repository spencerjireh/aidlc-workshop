"""Customer data ingestion and retrieval endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_customer_repo, get_segmentation_service
from src.api.schemas import CustomerIngestRequest
from src.models.service_models import DataFormat
from src.repositories.customer_repository import CustomerDataRepository
from src.services.segmentation_service import SegmentationService

router = APIRouter(prefix="/customers", tags=["customers"])


@router.post("/ingest")
def ingest_customer_data(
    request: CustomerIngestRequest,
    service: SegmentationService = Depends(get_segmentation_service),
):
    """Ingest customer data in JSON or CSV format."""
    fmt = DataFormat.JSON if request.format.lower() == "json" else DataFormat.CSV
    result = service.ingest_customer_data(request.customers, format=fmt)
    return result.model_dump()


@router.get("/{customer_id}")
def get_customer(
    customer_id: str,
    repo: CustomerDataRepository = Depends(get_customer_repo),
):
    """Retrieve a customer profile by ID."""
    customer = repo.get_customer(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer.model_dump()


@router.get("/{customer_id}/assignment")
def get_customer_assignment(
    customer_id: str,
    service: SegmentationService = Depends(get_segmentation_service),
):
    """Get a customer's segment assignment."""
    assignment = service.segment_repo.get_customer_assignment(customer_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    return assignment.model_dump()


@router.get("/{customer_id}/explanation")
def get_customer_explanation(
    customer_id: str,
    service: SegmentationService = Depends(get_segmentation_service),
):
    """Get explanation for a customer's segment assignment."""
    assignment = service.segment_repo.get_customer_assignment(customer_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")
    try:
        explanation = service.explain_assignment(customer_id, assignment.segment_id)
        return explanation.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
