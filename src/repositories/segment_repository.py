"""Segment data repository with versioning and assignment tracking."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
import json

from src.models.segment import Segment, CustomerSegmentAssignment, SegmentProfile


class SegmentDataRepository:
    """
    Repository for segments, cluster metadata, and customer-segment assignments.
    
    Uses in-memory storage for POC, designed to be extensible to PostgreSQL/MongoDB.
    
    Requirements:
    - 2.5: Customer-segment assignment storage
    - 10.5: Segment versioning for refinement tracking
    """
    
    def __init__(self):
        """Initialize repository with in-memory storage."""
        # In-memory storage (POC - extensible to real database)
        self._segments: Dict[str, Segment] = {}  # segment_id -> Segment
        self._assignments: Dict[str, CustomerSegmentAssignment] = {}  # assignment_id -> Assignment
        self._customer_to_segment: Dict[str, str] = {}  # customer_id -> segment_id
        self._segment_to_customers: Dict[str, List[str]] = {}  # segment_id -> list of customer_ids
        
        # Version history: segment_id -> list of historical versions
        self._segment_versions: Dict[str, List[Dict]] = {}
    
    def create_segment(self, segment: Segment) -> str:
        """
        Create a new segment.
        
        Args:
            segment: Segment to create
            
        Returns:
            Segment ID
        """
        self._segments[segment.segment_id] = segment
        self._segment_to_customers[segment.segment_id] = []
        
        # Initialize version history
        if segment.segment_id not in self._segment_versions:
            self._segment_versions[segment.segment_id] = []
        
        # Store initial version
        self._add_version_snapshot(segment)
        
        return segment.segment_id
    
    def get_segment(self, segment_id: str) -> Optional[Segment]:
        """
        Retrieve a segment by ID.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            Segment or None if not found
        """
        return self._segments.get(segment_id)
    
    def update_segment(self, segment_id: str, segment: Segment) -> bool:
        """
        Update an existing segment and track version history.
        
        Args:
            segment_id: Segment identifier
            segment: Updated segment
            
        Returns:
            True if updated, False if not found
        """
        if segment_id not in self._segments:
            return False
        
        # Store previous version before updating
        old_segment = self._segments[segment_id]
        self._add_version_snapshot(old_segment)
        
        # Update segment
        segment.updated_at = datetime.utcnow()
        segment.version = old_segment.version + 1
        self._segments[segment_id] = segment
        
        return True
    
    def delete_segment(self, segment_id: str) -> bool:
        """
        Delete a segment and its assignments.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            True if deleted, False if not found
        """
        if segment_id not in self._segments:
            return False
        
        # Remove all assignments for this segment
        customer_ids = self._segment_to_customers.get(segment_id, [])
        for customer_id in customer_ids:
            if customer_id in self._customer_to_segment:
                del self._customer_to_segment[customer_id]
        
        # Remove assignments
        assignments_to_remove = [
            aid for aid, assignment in self._assignments.items()
            if assignment.segment_id == segment_id
        ]
        for assignment_id in assignments_to_remove:
            del self._assignments[assignment_id]
        
        # Remove segment
        del self._segments[segment_id]
        del self._segment_to_customers[segment_id]
        
        return True
    
    def list_segments(self) -> List[Segment]:
        """
        List all segments.
        
        Returns:
            List of all segments
        """
        return list(self._segments.values())
    
    def assign_customer_to_segment(
        self, 
        assignment: CustomerSegmentAssignment
    ) -> str:
        """
        Assign a customer to a segment.
        
        Args:
            assignment: Customer segment assignment
            
        Returns:
            Assignment ID
        """
        # Store assignment
        self._assignments[assignment.assignment_id] = assignment
        
        # Update mappings
        self._customer_to_segment[assignment.customer_id] = assignment.segment_id
        
        if assignment.segment_id not in self._segment_to_customers:
            self._segment_to_customers[assignment.segment_id] = []
        
        if assignment.customer_id not in self._segment_to_customers[assignment.segment_id]:
            self._segment_to_customers[assignment.segment_id].append(assignment.customer_id)
        
        return assignment.assignment_id
    
    def get_customer_assignment(self, customer_id: str) -> Optional[CustomerSegmentAssignment]:
        """
        Get the segment assignment for a customer.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Customer segment assignment or None if not found
        """
        segment_id = self._customer_to_segment.get(customer_id)
        if not segment_id:
            return None
        
        # Find the assignment
        for assignment in self._assignments.values():
            if assignment.customer_id == customer_id and assignment.segment_id == segment_id:
                return assignment
        
        return None
    
    def get_segment_customers(self, segment_id: str) -> List[str]:
        """
        Get all customer IDs assigned to a segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            List of customer IDs
        """
        return self._segment_to_customers.get(segment_id, [])
    
    def get_segment_assignments(self, segment_id: str) -> List[CustomerSegmentAssignment]:
        """
        Get all assignments for a segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            List of customer segment assignments
        """
        return [
            assignment for assignment in self._assignments.values()
            if assignment.segment_id == segment_id
        ]
    
    def remove_customer_assignment(self, customer_id: str) -> bool:
        """
        Remove a customer's segment assignment.
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            True if removed, False if not found
        """
        segment_id = self._customer_to_segment.get(customer_id)
        if not segment_id:
            return False
        
        # Remove from mappings
        del self._customer_to_segment[customer_id]
        
        if segment_id in self._segment_to_customers:
            self._segment_to_customers[segment_id] = [
                cid for cid in self._segment_to_customers[segment_id]
                if cid != customer_id
            ]
        
        # Remove assignment
        assignment_to_remove = None
        for aid, assignment in self._assignments.items():
            if assignment.customer_id == customer_id and assignment.segment_id == segment_id:
                assignment_to_remove = aid
                break
        
        if assignment_to_remove:
            del self._assignments[assignment_to_remove]
            return True
        
        return False
    
    def get_segment_size(self, segment_id: str) -> int:
        """
        Get the number of customers in a segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            Number of customers
        """
        return len(self._segment_to_customers.get(segment_id, []))
    
    def count_segments(self) -> int:
        """
        Get total number of segments.
        
        Returns:
            Segment count
        """
        return len(self._segments)
    
    def segment_exists(self, segment_id: str) -> bool:
        """
        Check if a segment exists.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            True if segment exists
        """
        return segment_id in self._segments
    
    def _add_version_snapshot(self, segment: Segment) -> None:
        """
        Add a version snapshot to history.
        
        Args:
            segment: Segment to snapshot
        """
        snapshot = {
            'version': segment.version,
            'timestamp': segment.updated_at.isoformat(),
            'name': segment.name,
            'description': segment.description,
            'cluster_id': segment.cluster_id,
            'centroid': segment.centroid,
            'size': segment.size,
            'characteristics': segment.characteristics,
        }
        
        if segment.segment_id not in self._segment_versions:
            self._segment_versions[segment.segment_id] = []
        
        self._segment_versions[segment.segment_id].append(snapshot)
    
    def get_segment_version_history(self, segment_id: str) -> List[Dict]:
        """
        Get version history for a segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            List of version snapshots
        """
        return self._segment_versions.get(segment_id, [])
    
    def get_segment_version(self, segment_id: str, version: int) -> Optional[Dict]:
        """
        Get a specific version of a segment.
        
        Args:
            segment_id: Segment identifier
            version: Version number
            
        Returns:
            Version snapshot or None if not found
        """
        versions = self._segment_versions.get(segment_id, [])
        for snapshot in versions:
            if snapshot['version'] == version:
                return snapshot
        return None
    
    def rollback_segment_to_version(self, segment_id: str, version: int) -> bool:
        """
        Rollback a segment to a previous version.
        
        Args:
            segment_id: Segment identifier
            version: Version number to rollback to
            
        Returns:
            True if rolled back, False if version not found
        """
        snapshot = self.get_segment_version(segment_id, version)
        if not snapshot:
            return False
        
        current_segment = self._segments.get(segment_id)
        if not current_segment:
            return False
        
        # Create updated segment from snapshot
        current_segment.name = snapshot['name']
        current_segment.description = snapshot['description']
        current_segment.cluster_id = snapshot['cluster_id']
        current_segment.centroid = snapshot['centroid']
        current_segment.characteristics = snapshot['characteristics']
        current_segment.updated_at = datetime.utcnow()
        current_segment.version += 1
        
        # Add rollback to version history
        self._add_version_snapshot(current_segment)
        
        return True
    
    def clear_all_assignments(self) -> None:
        """
        Clear all customer-segment assignments.
        
        Useful for re-clustering operations.
        """
        self._assignments.clear()
        self._customer_to_segment.clear()
        for segment_id in self._segment_to_customers:
            self._segment_to_customers[segment_id] = []
    
    def get_assignment_by_id(self, assignment_id: str) -> Optional[CustomerSegmentAssignment]:
        """
        Get an assignment by its ID.
        
        Args:
            assignment_id: Assignment identifier
            
        Returns:
            Customer segment assignment or None if not found
        """
        return self._assignments.get(assignment_id)
