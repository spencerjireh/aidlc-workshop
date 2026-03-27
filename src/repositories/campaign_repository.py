"""Campaign data repository for campaigns, ad content, and performance metrics."""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from src.models.campaign import Campaign, AdContent, AdPerformanceMetrics, CampaignStatus


class CampaignDataRepository:
    """
    Repository for campaigns, ad content, and performance metrics.
    
    Uses in-memory storage for POC, designed to be extensible to PostgreSQL/MongoDB.
    
    Requirements:
    - 5.1: Campaign storage
    - 5.3: Campaign-segment associations
    - 7.2: Performance metrics storage
    """
    
    def __init__(self):
        """Initialize repository with in-memory storage."""
        # In-memory storage (POC - extensible to real database)
        self._campaigns: Dict[str, Campaign] = {}  # campaign_id -> Campaign
        self._ad_content: Dict[str, AdContent] = {}  # ad_id -> AdContent
        self._performance_metrics: Dict[str, AdPerformanceMetrics] = {}  # ad_id -> Metrics
        
        # Associations
        self._campaign_to_segments: Dict[str, List[str]] = {}  # campaign_id -> segment_ids
        self._segment_to_campaigns: Dict[str, List[str]] = {}  # segment_id -> campaign_ids
        self._campaign_to_ads: Dict[str, List[str]] = {}  # campaign_id -> ad_ids
    
    def create_campaign(self, campaign: Campaign) -> str:
        """
        Create a new campaign.
        
        Args:
            campaign: Campaign to create
            
        Returns:
            Campaign ID
        """
        self._campaigns[campaign.campaign_id] = campaign
        
        # Store segment associations
        self._campaign_to_segments[campaign.campaign_id] = campaign.target_segment_ids.copy()
        
        for segment_id in campaign.target_segment_ids:
            if segment_id not in self._segment_to_campaigns:
                self._segment_to_campaigns[segment_id] = []
            if campaign.campaign_id not in self._segment_to_campaigns[segment_id]:
                self._segment_to_campaigns[segment_id].append(campaign.campaign_id)
        
        # Store ad content associations
        self._campaign_to_ads[campaign.campaign_id] = campaign.ad_content_ids.copy()
        
        return campaign.campaign_id
    
    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """
        Retrieve a campaign by ID.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            Campaign or None if not found
        """
        return self._campaigns.get(campaign_id)
    
    def update_campaign(self, campaign_id: str, campaign: Campaign) -> bool:
        """
        Update an existing campaign.
        
        Args:
            campaign_id: Campaign identifier
            campaign: Updated campaign
            
        Returns:
            True if updated, False if not found
        """
        if campaign_id not in self._campaigns:
            return False
        
        # Update timestamp
        campaign.updated_at = datetime.utcnow()
        
        # Update campaign
        old_campaign = self._campaigns[campaign_id]
        self._campaigns[campaign_id] = campaign
        
        # Update segment associations if changed
        if old_campaign.target_segment_ids != campaign.target_segment_ids:
            # Remove old associations
            for segment_id in old_campaign.target_segment_ids:
                if segment_id in self._segment_to_campaigns:
                    self._segment_to_campaigns[segment_id] = [
                        cid for cid in self._segment_to_campaigns[segment_id]
                        if cid != campaign_id
                    ]
            
            # Add new associations
            self._campaign_to_segments[campaign_id] = campaign.target_segment_ids.copy()
            for segment_id in campaign.target_segment_ids:
                if segment_id not in self._segment_to_campaigns:
                    self._segment_to_campaigns[segment_id] = []
                if campaign_id not in self._segment_to_campaigns[segment_id]:
                    self._segment_to_campaigns[segment_id].append(campaign_id)
        
        # Update ad content associations if changed
        if old_campaign.ad_content_ids != campaign.ad_content_ids:
            self._campaign_to_ads[campaign_id] = campaign.ad_content_ids.copy()
        
        return True
    
    def delete_campaign(self, campaign_id: str) -> bool:
        """
        Delete a campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            True if deleted, False if not found
        """
        if campaign_id not in self._campaigns:
            return False
        
        campaign = self._campaigns[campaign_id]
        
        # Remove segment associations
        for segment_id in campaign.target_segment_ids:
            if segment_id in self._segment_to_campaigns:
                self._segment_to_campaigns[segment_id] = [
                    cid for cid in self._segment_to_campaigns[segment_id]
                    if cid != campaign_id
                ]
        
        # Remove campaign
        del self._campaigns[campaign_id]
        if campaign_id in self._campaign_to_segments:
            del self._campaign_to_segments[campaign_id]
        if campaign_id in self._campaign_to_ads:
            del self._campaign_to_ads[campaign_id]
        
        return True
    
    def list_campaigns(
        self, 
        status: Optional[CampaignStatus] = None
    ) -> List[Campaign]:
        """
        List campaigns, optionally filtered by status.
        
        Args:
            status: Optional campaign status filter
            
        Returns:
            List of campaigns
        """
        campaigns = list(self._campaigns.values())
        
        if status:
            campaigns = [c for c in campaigns if c.status == status]
        
        return campaigns
    
    def get_campaigns_by_segment(self, segment_id: str) -> List[Campaign]:
        """
        Get all campaigns targeting a specific segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            List of campaigns
        """
        campaign_ids = self._segment_to_campaigns.get(segment_id, [])
        return [self._campaigns[cid] for cid in campaign_ids if cid in self._campaigns]
    
    def create_ad_content(self, ad_content: AdContent) -> str:
        """
        Create ad content.
        
        Args:
            ad_content: Ad content to create
            
        Returns:
            Ad ID
        """
        self._ad_content[ad_content.ad_id] = ad_content
        return ad_content.ad_id
    
    def get_ad_content(self, ad_id: str) -> Optional[AdContent]:
        """
        Retrieve ad content by ID.
        
        Args:
            ad_id: Ad identifier
            
        Returns:
            Ad content or None if not found
        """
        return self._ad_content.get(ad_id)
    
    def update_ad_content(self, ad_id: str, ad_content: AdContent) -> bool:
        """
        Update ad content.
        
        Args:
            ad_id: Ad identifier
            ad_content: Updated ad content
            
        Returns:
            True if updated, False if not found
        """
        if ad_id not in self._ad_content:
            return False
        
        self._ad_content[ad_id] = ad_content
        return True
    
    def delete_ad_content(self, ad_id: str) -> bool:
        """
        Delete ad content.
        
        Args:
            ad_id: Ad identifier
            
        Returns:
            True if deleted, False if not found
        """
        if ad_id not in self._ad_content:
            return False
        
        del self._ad_content[ad_id]
        
        # Remove from performance metrics if exists
        if ad_id in self._performance_metrics:
            del self._performance_metrics[ad_id]
        
        return True
    
    def get_campaign_ads(self, campaign_id: str) -> List[AdContent]:
        """
        Get all ad content for a campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            List of ad content
        """
        ad_ids = self._campaign_to_ads.get(campaign_id, [])
        return [self._ad_content[aid] for aid in ad_ids if aid in self._ad_content]
    
    def get_segment_ads(self, segment_id: str) -> List[AdContent]:
        """
        Get all ad content targeting a segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            List of ad content
        """
        return [
            ad for ad in self._ad_content.values()
            if ad.segment_id == segment_id
        ]
    
    def store_performance_metrics(self, metrics: AdPerformanceMetrics) -> str:
        """
        Store or update performance metrics for an ad.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Ad ID
        """
        self._performance_metrics[metrics.ad_id] = metrics
        
        # Update ad content with metrics
        if metrics.ad_id in self._ad_content:
            ad = self._ad_content[metrics.ad_id]
            ad.performance_metrics = metrics
        
        return metrics.ad_id
    
    def get_performance_metrics(self, ad_id: str) -> Optional[AdPerformanceMetrics]:
        """
        Get performance metrics for an ad.
        
        Args:
            ad_id: Ad identifier
            
        Returns:
            Performance metrics or None if not found
        """
        return self._performance_metrics.get(ad_id)
    
    def get_campaign_performance(self, campaign_id: str) -> List[AdPerformanceMetrics]:
        """
        Get performance metrics for all ads in a campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            List of performance metrics
        """
        ad_ids = self._campaign_to_ads.get(campaign_id, [])
        metrics = []
        
        for ad_id in ad_ids:
            if ad_id in self._performance_metrics:
                metrics.append(self._performance_metrics[ad_id])
        
        return metrics
    
    def get_segment_performance(self, segment_id: str) -> List[AdPerformanceMetrics]:
        """
        Get performance metrics for all ads targeting a segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            List of performance metrics
        """
        return [
            metrics for metrics in self._performance_metrics.values()
            if metrics.segment_id == segment_id
        ]
    
    def count_campaigns(self) -> int:
        """
        Get total number of campaigns.
        
        Returns:
            Campaign count
        """
        return len(self._campaigns)
    
    def count_ads(self) -> int:
        """
        Get total number of ad content items.
        
        Returns:
            Ad content count
        """
        return len(self._ad_content)
    
    def campaign_exists(self, campaign_id: str) -> bool:
        """
        Check if a campaign exists.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            True if campaign exists
        """
        return campaign_id in self._campaigns
    
    def ad_exists(self, ad_id: str) -> bool:
        """
        Check if ad content exists.
        
        Args:
            ad_id: Ad identifier
            
        Returns:
            True if ad exists
        """
        return ad_id in self._ad_content
