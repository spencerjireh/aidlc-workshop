"""Data repositories for customer, segment, and campaign data."""

from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.repositories.campaign_repository import CampaignDataRepository

__all__ = [
    "CustomerDataRepository",
    "SegmentDataRepository",
    "CampaignDataRepository",
]
