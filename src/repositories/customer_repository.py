"""Customer data repository with PII anonymization."""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional

from src.models.customer import CustomerProfile, TransactionData


class CustomerDataRepository:
    """
    Repository for customer data with PII anonymization.

    Uses in-memory storage for POC, designed to be extensible to PostgreSQL/MongoDB.

    Requirements:
    - 1.1: Customer data ingestion
    - 8.1: PII anonymization before storage
    """

    def __init__(self):
        """Initialize repository with in-memory storage."""
        # In-memory storage (POC - extensible to real database)
        self._customers: Dict[str, str] = {}  # customer_id -> JSON data
        self._transactions: Dict[str, List[str]] = {}  # customer_id -> list of JSON transactions

    def _anonymize_customer_id(self, original_id: str) -> str:
        """
        Anonymize customer ID using SHA-256 hash (non-reversible).

        Args:
            original_id: Original customer identifier

        Returns:
            Anonymized customer ID (SHA-256 hash)
        """
        return hashlib.sha256(original_id.encode()).hexdigest()[:16]

    def create_customer(self, customer: CustomerProfile) -> str:
        """
        Create a new customer profile with PII anonymization.

        Args:
            customer: Customer profile to store

        Returns:
            Anonymized customer ID
        """
        # Anonymize customer ID
        anonymized_id = self._anonymize_customer_id(customer.customer_id)

        # Create a copy with anonymized ID
        customer_data = customer.model_dump()
        customer_data['customer_id'] = anonymized_id

        # Store as JSON
        self._customers[anonymized_id] = json.dumps(customer_data, default=str)

        # Initialize transaction list
        if anonymized_id not in self._transactions:
            self._transactions[anonymized_id] = []

        return anonymized_id

    def get_customer(self, customer_id: str) -> Optional[CustomerProfile]:
        """
        Retrieve a customer profile by ID.

        Args:
            customer_id: Customer identifier (can be original or anonymized)

        Returns:
            Customer profile or None if not found
        """
        # Try as-is first, then try anonymizing
        anonymized_id = customer_id
        if customer_id not in self._customers:
            anonymized_id = self._anonymize_customer_id(customer_id)

        if anonymized_id not in self._customers:
            return None

        # Deserialize
        customer_data = json.loads(self._customers[anonymized_id])

        return CustomerProfile(**customer_data)

    def update_customer(self, customer_id: str, customer: CustomerProfile) -> bool:
        """
        Update an existing customer profile.

        Args:
            customer_id: Customer identifier (can be original or anonymized)
            customer: Updated customer profile

        Returns:
            True if updated, False if not found
        """
        # Try as-is first, then try anonymizing
        anonymized_id = customer_id
        if customer_id not in self._customers:
            anonymized_id = self._anonymize_customer_id(customer_id)

        if anonymized_id not in self._customers:
            return False

        # Update timestamp
        customer_data = customer.model_dump()
        customer_data['customer_id'] = anonymized_id
        customer_data['updated_at'] = datetime.utcnow()

        # Store as JSON
        self._customers[anonymized_id] = json.dumps(customer_data, default=str)

        return True

    def delete_customer(self, customer_id: str) -> bool:
        """
        Delete a customer profile.

        Args:
            customer_id: Customer identifier (can be original or anonymized)

        Returns:
            True if deleted, False if not found
        """
        # Try as-is first, then try anonymizing
        anonymized_id = customer_id
        if customer_id not in self._customers:
            anonymized_id = self._anonymize_customer_id(customer_id)

        if anonymized_id not in self._customers:
            return False

        del self._customers[anonymized_id]
        if anonymized_id in self._transactions:
            del self._transactions[anonymized_id]

        return True

    def list_customers(self, limit: Optional[int] = None, offset: int = 0) -> List[CustomerProfile]:
        """
        List all customer profiles with pagination.

        Args:
            limit: Maximum number of customers to return
            offset: Number of customers to skip

        Returns:
            List of customer profiles
        """
        customer_ids = list(self._customers.keys())[offset:]
        if limit:
            customer_ids = customer_ids[:limit]

        customers = []
        for customer_id in customer_ids:
            customer = self.get_customer(customer_id)
            if customer:
                customers.append(customer)

        return customers

    def add_transaction(self, transaction: TransactionData) -> str:
        """
        Add a transaction for a customer.

        Args:
            transaction: Transaction data

        Returns:
            Transaction ID
        """
        # Anonymize customer ID
        anonymized_customer_id = self._anonymize_customer_id(transaction.customer_id)

        # Create transaction data with anonymized customer ID
        transaction_data = transaction.model_dump()
        transaction_data['customer_id'] = anonymized_customer_id

        # Store as JSON
        json_data = json.dumps(transaction_data, default=str)

        if anonymized_customer_id not in self._transactions:
            self._transactions[anonymized_customer_id] = []

        self._transactions[anonymized_customer_id].append(json_data)

        return transaction.transaction_id

    def get_customer_transactions(self, customer_id: str) -> List[TransactionData]:
        """
        Get all transactions for a customer.

        Args:
            customer_id: Customer identifier

        Returns:
            List of transactions
        """
        anonymized_id = self._anonymize_customer_id(customer_id)

        if anonymized_id not in self._transactions:
            return []

        transactions = []
        for json_data in self._transactions[anonymized_id]:
            transaction_data = json.loads(json_data)
            transactions.append(TransactionData(**transaction_data))

        return transactions

    def count_customers(self) -> int:
        """
        Get total number of customers.

        Returns:
            Customer count
        """
        return len(self._customers)

    def customer_exists(self, customer_id: str) -> bool:
        """
        Check if a customer exists.

        Args:
            customer_id: Customer identifier

        Returns:
            True if customer exists
        """
        anonymized_id = self._anonymize_customer_id(customer_id)
        return anonymized_id in self._customers
