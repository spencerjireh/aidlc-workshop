"""Customer-related data models."""

from datetime import datetime
from typing import List
from pydantic import BaseModel, Field, field_validator


class TransactionData(BaseModel):
    """Historical transaction record for a customer."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    customer_id: str = Field(..., description="References CustomerProfile")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., description="Category of merchant")
    merchant_name: str = Field(..., description="Name of merchant")
    transaction_type: str = Field(..., description="Type: payment, transfer, cashout")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    payment_method: str = Field(..., description="Payment method used")
    location: str = Field(..., description="Transaction location")
    
    @field_validator('transaction_type')
    @classmethod
    def validate_transaction_type(cls, v: str) -> str:
        """Validate transaction type is one of the allowed values."""
        allowed_types = {'payment', 'transfer', 'cashout'}
        if v not in allowed_types:
            raise ValueError(f"transaction_type must be one of {allowed_types}")
        return v


class CustomerProfile(BaseModel):
    """Customer profile with demographics and behavioral attributes."""
    
    customer_id: str = Field(..., description="Unique identifier (anonymized)")
    age: int = Field(..., ge=0, le=120, description="Customer age")
    location: str = Field(..., description="Geographic location (city/region)")
    transaction_frequency: int = Field(..., ge=0, description="Number of transactions per month")
    average_transaction_value: float = Field(..., ge=0, description="Average transaction amount")
    merchant_categories: List[str] = Field(..., description="Top merchant categories")
    total_spend: float = Field(..., ge=0, description="Total spending amount")
    account_age_days: int = Field(..., ge=0, description="Days since account creation")
    preferred_payment_methods: List[str] = Field(..., description="Payment methods used")
    last_transaction_date: datetime = Field(..., description="Most recent transaction")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Profile update timestamp")
    
    @field_validator('merchant_categories', 'preferred_payment_methods')
    @classmethod
    def validate_non_empty_list(cls, v: List[str]) -> List[str]:
        """Ensure lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v
