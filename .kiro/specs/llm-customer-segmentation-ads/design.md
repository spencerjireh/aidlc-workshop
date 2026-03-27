# Design Document: ML-Powered Customer Segmentation Advertising System

## Overview

This document provides the technical design for a proof-of-concept system that uses machine learning techniques (Principal Component Analysis and K-Means Clustering) combined with Large Language Models (LLMs) to deliver targeted advertising for e-wallet platforms. The system analyzes customer transaction data, behavioral patterns, and demographic information using PCA for dimensionality reduction and K-Means for clustering, while leveraging LLMs for content generation and natural language insights.

### System Goals

- Automatically segment customers using PCA and K-Means clustering on transaction and behavioral data
- Use LLMs to generate natural language insights and explanations for segments
- Create personalized ad content tailored to segment characteristics
- Provide explainability for segmentation decisions through cluster analysis and LLM interpretation
- Enable interactive querying of segmentation data through a conversational chatbot interface
- Maintain basic PII anonymization (full security deferred to production)

### Key Design Principles

1. **ML-First Segmentation**: Use proven ML algorithms (PCA + K-Means) for data-driven clustering
2. **LLM for Interpretation**: LLMs generate human-readable insights from cluster statistics
3. **Provider Agnostic**: Support multiple LLM providers through abstraction layer
4. **Basic PII Handling**: Anonymize PII before processing (encryption and auth deferred to production)
5. **Explainability**: Segmentation decisions explained through cluster centroids, feature importance, and LLM narratives
6. **Scalability**: Handle 10,000+ customer records efficiently
7. **Conversational Analytics**: Enable natural language queries for data exploration

## Architecture

### High-Level Architecture

The system follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│              Analytics Dashboard (POC Scope)                     │
│  (Segment Visualization, Campaign Management, Query Chatbot)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    Application Services Layer                    │
├──────────────────┬──────────────────┬──────────────────────────┤
│ Segmentation     │  Ad Generator    │  Targeting Engine        │
│ Service          │  Service         │  Service                 │
└──────────────────┴──────────────────┴──────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                   ML & LLM Engine Layer                          │
├──────────────────┬──────────────────┬──────────────────────────┤
│  PCA Engine      │  K-Means Engine  │  LLM Engine              │
│ (Dimensionality) │  (Clustering)    │ (Content & Insights)     │
└──────────────────┴──────────────────┴──────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    LLM Provider Adapters                         │
├──────────────────┬──────────────────┬──────────────────────────┤
│  OpenAI Adapter  │ Anthropic Adapter│  Local Model Adapter     │
└──────────────────┴──────────────────┴──────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                      Data Layer                                  │
├──────────────────┬──────────────────┬──────────────────────────┤
│ Customer Data    │  Segment Data    │  Campaign Data           │
│ Repository       │  Repository      │  Repository              │
└──────────────────┴──────────────────┴──────────────────────────┘
```

### Component Interaction Flow

```
User Request (Dashboard/Chatbot)
    │
    ├─→ Segmentation Service
    │       ├─→ Data Repository (fetch customer data)
    │       ├─→ PCA Engine (reduce dimensionality)
    │       ├─→ K-Means Engine (cluster customers)
    │       ├─→ LLM Engine (generate segment names & descriptions)
    │       └─→ Data Repository (store segments)
    │
    ├─→ Ad Generator Service
    │       ├─→ Segment Repository (fetch segment profile)
    │       ├─→ LLM Engine (generate ad content)
    │       └─→ Campaign Repository (store ad content)
    │
    ├─→ Targeting Engine
    │       ├─→ Segment Repository (fetch segment data)
    │       ├─→ Campaign Repository (associate campaigns)
    │       └─→ Analytics Repository (track performance)
    │
    └─→ Query Chatbot
            ├─→ LLM Engine (interpret query)
            ├─→ Data Repositories (fetch relevant data)
            ├─→ LLM Engine (generate response)
            └─→ Dashboard (display response)
```

### Technology Stack Considerations

**Backend Services**:
- Python for service implementation (scikit-learn for PCA and K-Means)
- FastAPI for REST API endpoints
- Async processing for LLM calls to handle latency

**ML Libraries**:
- scikit-learn for PCA and K-Means clustering
- NumPy and Pandas for data processing
- Matplotlib/Seaborn for cluster visualization

**LLM Integration**:
- OpenAI SDK for GPT models
- Anthropic SDK for Claude models
- LangChain or LlamaIndex for LLM orchestration and prompt management
- Local model support via Ollama or vLLM

**Data Storage** (POC):
- In-memory storage for customer, segment, and campaign data (extensible to PostgreSQL/MongoDB in production)

**Analytics Dashboard** (POC):
- React or Vue.js for frontend
- Chart.js or D3.js for visualizations
- REST polling for chatbot interactions

> **Deferred to production:** Redis caching, vector databases (Pinecone/Weaviate), AES-256 encryption, TLS 1.3, JWT authentication, RBAC, WebSocket for real-time chatbot.



## Components and Interfaces

### 1. Segmentation Service

**Responsibilities**:
- Ingest and validate customer data
- Normalize and prepare features for PCA
- Coordinate with PCA Engine for dimensionality reduction
- Coordinate with K-Means Engine for clustering
- Coordinate with LLM Engine to generate segment names and descriptions
- Assign customers to segments with distance-based confidence scores
- Provide segment refinement capabilities through re-clustering

**Key Methods**:

```python
class SegmentationService:
    def ingest_customer_data(
        self, 
        data: List[CustomerProfile], 
        format: DataFormat
    ) -> IngestionResult:
        """
        Parse and store customer data from JSON/CSV format.
        Validates data, handles duplicates, logs errors.
        Returns: IngestionResult with success/failure counts
        """
        
    def create_segments(
        self, 
        customer_ids: List[str],
        num_clusters: Optional[int] = None
    ) -> List[Segment]:
        """
        Create segments using PCA + K-Means clustering.
        Steps:
        1. Extract and normalize customer features
        2. Apply PCA for dimensionality reduction (80% variance)
        3. Determine optimal k using Elbow Method or Silhouette Score (3-10 clusters)
        4. Run K-Means clustering on PCA-transformed data
        5. Use LLM to generate descriptive names and profiles for each cluster
        Returns: List of Segment objects with names, descriptions, cluster statistics
        """
        
    def assign_customers_to_segments(
        self,
        customer_ids: List[str],
        segments: List[Segment]
    ) -> List[CustomerSegmentAssignment]:
        """
        Assign each customer to exactly one segment based on cluster membership.
        Confidence score calculated from distance to cluster centroid (normalized).
        Returns: List of assignments with confidence scores (0-1)
        """
        
    def explain_assignment(
        self,
        customer_id: str,
        segment_id: str
    ) -> AssignmentExplanation:
        """
        Generate explanation for customer-segment assignment.
        Uses PCA loadings to identify top contributing features.
        LLM generates natural language explanation from feature values.
        Returns: Explanation with top 3 contributing features and their importance
        """
        
    def refine_segment(
        self,
        num_clusters: int
    ) -> List[Segment]:
        """
        Re-cluster customers with new k value.
        Re-assigns all customers and tracks version history.
        Returns: Updated list of Segment objects
        """
```

**Interfaces**:
- REST API endpoints for data ingestion and segment management
- Event-driven interface for segment updates (pub/sub pattern)

### 2. PCA Engine

**Responsibilities**:
- Normalize customer features for PCA
- Perform Principal Component Analysis for dimensionality reduction
- Determine number of components to retain (80% variance explained)
- Provide feature loadings for explainability

**Key Methods**:

```python
class PCAEngine:
    def fit_transform(
        self,
        customer_features: np.ndarray,
        variance_threshold: float = 0.8
    ) -> PCAResult:
        """
        Apply PCA to customer features.
        Retains components explaining at least variance_threshold of total variance.
        Returns: PCAResult with transformed data, explained variance, and loadings
        """
        
    def get_feature_importance(
        self,
        component_idx: int
    ) -> List[Tuple[str, float]]:
        """
        Get feature loadings for a specific principal component.
        Returns: List of (feature_name, loading) tuples sorted by absolute value
        """
        
    def transform(
        self,
        customer_features: np.ndarray
    ) -> np.ndarray:
        """
        Transform new customer data using fitted PCA model.
        Returns: Transformed feature matrix
        """
```

### 3. K-Means Engine

**Responsibilities**:
- Determine optimal number of clusters using Elbow Method or Silhouette Score
- Perform K-Means clustering on PCA-transformed data
- Calculate cluster centroids and statistics
- Compute distance-based confidence scores for assignments

**Key Methods**:

```python
class KMeansEngine:
    def determine_optimal_k(
        self,
        pca_data: np.ndarray,
        k_range: Tuple[int, int] = (3, 10)
    ) -> int:
        """
        Determine optimal number of clusters using Elbow Method and Silhouette Score.
        Tests k values in range and selects best based on combined metrics.
        Returns: Optimal k value
        """
        
    def fit_predict(
        self,
        pca_data: np.ndarray,
        n_clusters: int
    ) -> ClusteringResult:
        """
        Perform K-Means clustering on PCA-transformed data.
        Returns: ClusteringResult with cluster labels, centroids, and inertia
        """
        
    def calculate_confidence_score(
        self,
        customer_point: np.ndarray,
        assigned_cluster: int
    ) -> float:
        """
        Calculate confidence score based on distance to cluster centroid.
        Normalized to range [0.0, 1.0] where 1.0 is at centroid.
        Returns: Confidence score
        """
        
    def get_cluster_statistics(
        self,
        cluster_id: int,
        customer_data: List[CustomerProfile]
    ) -> ClusterStatistics:
        """
        Calculate statistics for customers in a cluster.
        Returns: ClusterStatistics with size, averages, distributions
        """
```

### 4. LLM Engine

**Responsibilities**:
- Generate descriptive names and profiles for clusters based on statistics
- Generate natural language explanations for customer assignments
- Generate personalized ad content
- Interpret and respond to chatbot queries
- Abstract multiple LLM provider implementations
- Handle API calls with retry logic and error handling

**Key Methods**:

```python
class LLMEngine:
    def generate_segment_profile(
        self,
        cluster_statistics: ClusterStatistics,
        cluster_id: int
    ) -> SegmentProfile:
        """
        Generate descriptive name and natural language profile for a cluster.
        Uses cluster statistics (demographics, behaviors, top categories) as input.
        Returns: SegmentProfile with name, description, and differentiating factors
        """
        
    def explain_cluster_assignment(
        self,
        customer: CustomerProfile,
        cluster_centroid: np.ndarray,
        top_features: List[Tuple[str, float]]
    ) -> str:
        """
        Generate natural language explanation for why customer was assigned to cluster.
        Uses customer features and PCA loadings to identify key factors.
        Returns: Natural language explanation string
        """
        
    def generate_ad_content(
        self,
        segment_profile: SegmentProfile,
        campaign_brief: str,
        format: AdFormat
    ) -> List[AdContent]:
        """
        Generate personalized ad copy for a segment.
        Returns: List of AdContent variations (minimum 3 for A/B testing)
        """
        
    def interpret_query(
        self,
        query: str,
        context: ConversationContext
    ) -> QueryIntent:
        """
        Interpret natural language query from chatbot.
        Returns: QueryIntent with structured query parameters
        """
        
    def generate_response(
        self,
        query_intent: QueryIntent,
        data: Dict[str, Any],
        context: ConversationContext
    ) -> ChatbotResponse:
        """
        Generate natural language response with supporting data.
        Returns: ChatbotResponse with text and optional visualizations
        """
        
    def call_llm(
        self,
        prompt: str,
        provider: LLMProvider,
        parameters: LLMParameters
    ) -> LLMResponse:
        """
        Low-level method to call LLM provider with retry logic.
        Implements exponential backoff (3 retries).
        Returns: LLMResponse with generated text and metadata
        """
```

**Provider Adapter Interface**:

```python
class LLMProviderAdapter(ABC):
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate API credentials and connection"""
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        parameters: LLMParameters
    ) -> str:
        """Generate text from prompt"""
        
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name (OpenAI, Anthropic, Local)"""
```

### 5. Ad Generator Service

**Responsibilities**:
- Generate personalized ad content for segments
- Create multiple ad variations for A/B testing
- Filter inappropriate content
- Support multiple ad formats (short, medium, long)

**Key Methods**:

```python
class AdGeneratorService:
    def create_campaign_ads(
        self,
        segment_id: str,
        campaign_brief: str,
        formats: List[AdFormat]
    ) -> CampaignAdSet:
        """
        Generate ad content for a campaign targeting a specific segment.
        Creates 3+ variations per format for A/B testing.
        Returns: CampaignAdSet with all ad variations
        """
        
    def validate_ad_content(
        self,
        ad_content: str
    ) -> ContentValidationResult:
        """
        Check for inappropriate or sensitive content.
        Returns: ValidationResult with approval status and issues
        """
```

### 6. Targeting Engine

**Responsibilities**:
- Manage campaign targeting to segments
- Calculate estimated reach
- Enforce minimum segment size constraints
- Track campaign-segment associations

**Key Methods**:

```python
class TargetingEngine:
    def create_campaign(
        self,
        campaign_name: str,
        target_segment_ids: List[str],
        ad_content: CampaignAdSet
    ) -> Campaign:
        """
        Create campaign targeting specific segments.
        Validates segment sizes (minimum 100 customers).
        Returns: Campaign object with estimated reach
        """
        
    def calculate_reach(
        self,
        segment_ids: List[str]
    ) -> ReachEstimate:
        """
        Calculate estimated reach for segment selection.
        Returns: ReachEstimate with total customers and breakdown by segment
        """
        
    def activate_campaign(
        self,
        campaign_id: str
    ) -> CampaignActivationResult:
        """
        Activate campaign and associate ad content with target segments.
        Returns: Activation result with status
        """
```

### 7. Query Chatbot Service

**Responsibilities**:
- Provide conversational interface for data queries
- Maintain conversation context across interactions
- Interpret natural language questions
- Generate responses with supporting data and visualizations
- Handle query limitations gracefully

**Key Methods**:

```python
class QueryChatbotService:
    def process_query(
        self,
        query: str,
        session_id: str
    ) -> ChatbotResponse:
        """
        Process natural language query and generate response.
        Maintains conversation context within session.
        Returns: ChatbotResponse with answer and optional visualizations
        """
        
    def get_conversation_context(
        self,
        session_id: str
    ) -> ConversationContext:
        """
        Retrieve conversation history for context-aware responses.
        Returns: ConversationContext with previous queries and responses
        """
        
    def suggest_alternative_queries(
        self,
        failed_query: str
    ) -> List[str]:
        """
        Suggest alternative questions when query cannot be answered.
        Returns: List of suggested queries
        """
```

### 8. Analytics Dashboard (POC Scope)

**Responsibilities**:
- Display segment visualizations and statistics
- Provide campaign management interface
- Host Query Chatbot interface

**Key Features** (POC):
- Segment distribution charts (pie, bar charts)
- PCA variance explained visualization
- Cluster centroid comparison views
- Campaign creation, listing, and activation interface
- Interactive chatbot panel with query input and response display

> **Deferred to production:** Customer assignment explorer, analytics/reporting interface with time-range filtering, LLM configuration UI, CSV/PDF export, segment comparison views.



## Data Models

### Customer Profile

```python
@dataclass
class CustomerProfile:
    customer_id: str                    # Unique identifier (anonymized)
    age: int                            # Customer age
    location: str                       # Geographic location (city/region)
    transaction_frequency: int          # Number of transactions per month
    average_transaction_value: float    # Average transaction amount
    merchant_categories: List[str]      # Top merchant categories (e.g., "Food", "Transport")
    total_spend: float                  # Total spending amount
    account_age_days: int               # Days since account creation
    preferred_payment_methods: List[str] # Payment methods used
    last_transaction_date: datetime     # Most recent transaction
    created_at: datetime
    updated_at: datetime
```

### Transaction Data

```python
@dataclass
class TransactionData:
    transaction_id: str
    customer_id: str                    # References CustomerProfile
    amount: float
    merchant_category: str
    merchant_name: str
    transaction_type: str               # "payment", "transfer", "cashout"
    timestamp: datetime
    payment_method: str
    location: str
```

### Segment

```python
@dataclass
class Segment:
    segment_id: str
    name: str                           # LLM-generated descriptive name
    description: str                    # Natural language profile summary
    characteristics: Dict[str, Any]     # Key characteristics (demographics, behaviors)
    cluster_id: int                     # K-Means cluster ID
    centroid: np.ndarray                # Cluster centroid in PCA space
    size: int                           # Number of customers in segment
    average_transaction_value: float
    transaction_frequency: float
    top_merchant_categories: List[str]
    differentiating_factors: List[str]  # What makes this segment unique
    pca_component_contributions: Dict[int, float]  # PC index -> contribution
    created_at: datetime
    updated_at: datetime
    version: int                        # For segment refinement tracking
```

### Customer Segment Assignment

```python
@dataclass
class CustomerSegmentAssignment:
    assignment_id: str
    customer_id: str
    segment_id: str
    confidence_score: float             # 0.0 to 1.0 (based on distance to centroid)
    distance_to_centroid: float         # Euclidean distance in PCA space
    assigned_at: datetime
    explanation: Optional[str]          # Natural language explanation
    contributing_factors: List[ContributingFactor]
```

### Contributing Factor

```python
@dataclass
class ContributingFactor:
    factor_name: str                    # e.g., "High transaction frequency"
    importance: float                   # Relative importance from PCA loadings (0.0 to 1.0)
    data_point: str                     # Specific data reference
    pca_loading: Optional[float]        # PCA loading value for this feature
```

### PCA Result

```python
@dataclass
class PCAResult:
    transformed_data: np.ndarray        # Customer data in PCA space
    explained_variance: List[float]     # Variance explained by each component
    explained_variance_ratio: List[float]  # Ratio of variance explained
    components: np.ndarray              # Principal component vectors
    feature_names: List[str]            # Original feature names
    n_components: int                   # Number of components retained
```

### Clustering Result

```python
@dataclass
class ClusteringResult:
    cluster_labels: np.ndarray          # Cluster assignment for each customer
    centroids: np.ndarray               # Cluster centroids in PCA space
    inertia: float                      # Sum of squared distances to centroids
    n_clusters: int                     # Number of clusters
    silhouette_score: float             # Clustering quality metric
```

### Cluster Statistics

```python
@dataclass
class ClusterStatistics:
    cluster_id: int
    size: int
    average_age: float
    location_distribution: Dict[str, int]
    average_transaction_frequency: float
    average_transaction_value: float
    total_spend_distribution: Dict[str, float]  # Percentiles
    top_merchant_categories: List[Tuple[str, int]]  # (category, count)
    preferred_payment_methods: Dict[str, int]
```

### Ad Content

```python
@dataclass
class AdContent:
    ad_id: str
    segment_id: str
    campaign_id: str
    format: AdFormat                    # SHORT (50 chars), MEDIUM (150), LONG (300)
    content: str                        # The ad copy
    variation_number: int               # For A/B testing (1, 2, 3, etc.)
    use_case: str                       # "cashback", "merchant_promo", "payment_convenience"
    created_at: datetime
    performance_metrics: Optional[AdPerformanceMetrics]
```

### Ad Format

```python
class AdFormat(Enum):
    SHORT = "short"      # 50 characters
    MEDIUM = "medium"    # 150 characters
    LONG = "long"        # 300 characters
```

### Campaign

```python
@dataclass
class Campaign:
    campaign_id: str
    name: str
    target_segment_ids: List[str]
    ad_content_ids: List[str]
    status: CampaignStatus              # DRAFT, ACTIVE, PAUSED, COMPLETED
    estimated_reach: int
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime
```

### Campaign Status

```python
class CampaignStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
```

### Ad Performance Metrics

```python
@dataclass
class AdPerformanceMetrics:
    ad_id: str
    impressions: int
    clicks: int
    conversions: int
    click_through_rate: float           # clicks / impressions
    conversion_rate: float              # conversions / clicks
    segment_id: str
    measurement_period_start: datetime
    measurement_period_end: datetime
```

### LLM Configuration

```python
@dataclass
class LLMConfiguration:
    config_id: str
    provider: LLMProvider               # OPENAI, ANTHROPIC, LOCAL
    model_name: str                     # e.g., "gpt-4", "claude-3-opus"
    api_key: str                        # Encrypted
    api_endpoint: Optional[str]         # For local models
    parameters: LLMParameters
    is_active: bool
    created_at: datetime
```

### LLM Provider

```python
class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
```

### LLM Parameters

```python
@dataclass
class LLMParameters:
    temperature: float = 0.7            # 0.0 to 2.0
    max_tokens: int = 1000
    top_p: float = 1.0                  # 0.0 to 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
```

### Conversation Context

```python
@dataclass
class ConversationContext:
    session_id: str
    user_id: str
    conversation_history: List[ChatMessage]
    created_at: datetime
    last_interaction: datetime
    metadata: Dict[str, Any]            # Additional context data
```

### Chat Message

```python
@dataclass
class ChatMessage:
    message_id: str
    role: MessageRole                   # USER, ASSISTANT
    content: str
    timestamp: datetime
    query_intent: Optional[QueryIntent]
    data_references: List[str]          # IDs of data used in response
```

### Message Role

```python
class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
```

### Query Intent

```python
@dataclass
class QueryIntent:
    intent_type: QueryType              # SEGMENT_INFO, COMPARISON, PERFORMANCE, TREND
    entities: Dict[str, Any]            # Extracted entities (segment names, dates, metrics)
    filters: Dict[str, Any]             # Query filters
    confidence: float                   # Intent classification confidence
```

### Query Type

```python
class QueryType(Enum):
    SEGMENT_INFO = "segment_info"           # "Tell me about segment X"
    COMPARISON = "comparison"               # "Compare segment A and B"
    PERFORMANCE = "performance"             # "How is campaign X performing?"
    TREND = "trend"                         # "Show trends over time"
    CUSTOMER_COUNT = "customer_count"       # "How many customers in segment X?"
    TOP_CATEGORIES = "top_categories"       # "What are top categories for segment X?"
```

### Chatbot Response

```python
@dataclass
class ChatbotResponse:
    response_id: str
    session_id: str
    text: str                           # Natural language response
    data: Optional[Dict[str, Any]]      # Supporting data for visualizations
    visualization_type: Optional[str]   # "table", "chart", "list"
    suggested_followups: List[str]      # Suggested follow-up questions
    response_time_ms: int
    timestamp: datetime
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Data Ingestion Round-Trip

*For any* valid customer data in JSON or CSV format, parsing and then serializing the data should produce an equivalent representation with all fields preserved.

**Validates: Requirements 1.1, 1.2**

### Property 2: Error Isolation in Batch Processing

*For any* dataset containing a mix of valid and invalid customer records, the system should process all valid records successfully while logging errors for invalid records, and the count of processed records plus error count should equal the total input count.

**Validates: Requirements 1.3**

### Property 3: Duplicate Merge Uses Latest Data

*For any* set of duplicate customer records with different timestamps, merging should result in a single record where all fields match the record with the most recent timestamp.

**Validates: Requirements 1.5**

### Property 4: Segment Count Bounds

*For any* customer dataset, the number of segments created using K-Means should be between 3 and 10 (inclusive), as determined by the Elbow Method or Silhouette Score.

**Validates: Requirements 2.2, 2.3**

### Property 5: Segment Completeness

*For any* generated segment, it must have a non-empty LLM-generated name, a non-empty description containing references to cluster statistics (demographics, spending behaviors, merchant categories), cluster centroid coordinates, and a non-empty list of differentiating factors.

**Validates: Requirements 2.4, 3.1, 3.3**

### Property 6: Unique Segment Assignment

*For any* set of customers and K-Means clusters, each customer must be assigned to exactly one segment based on their cluster membership (no customer appears in multiple segments, and no customer is unassigned).

**Validates: Requirements 2.5**

### Property 7: Confidence Score Bounds

*For any* customer-segment assignment, the confidence score (calculated from normalized distance to cluster centroid) must be in the range [0.0, 1.0] inclusive.

**Validates: Requirements 2.6**

### Property 8: Segment Statistics Consistency

*For any* segment, the displayed segment size must equal the count of customers assigned to that segment, and the calculated average transaction value must equal the sum of all customer transaction values divided by the segment size.

**Validates: Requirements 3.2**

### Property 9: Segment Profile Updates on Data Changes

*For any* segment, when new customer data is ingested and re-clustering is performed, the segment profile statistics (size, average transaction value, transaction frequency, cluster centroids) must be recalculated.

**Validates: Requirements 3.5**

### Property 9a: PCA Variance Threshold

*For any* PCA transformation, the number of principal components retained must explain at least 80% of the total variance in the customer feature data.

**Validates: Requirements 2.1**

### Property 9b: K-Means Cluster Assignment Determinism

*For any* customer and fitted K-Means model, running cluster assignment multiple times on the same customer data must produce the same cluster labels.

**Validates: Requirements 2.5**

### Property 9c: Distance-Based Confidence Score

*For any* customer-segment assignment, the confidence score must be inversely related to the distance to the cluster centroid (customers closer to centroid have higher confidence scores).

**Validates: Requirements 2.6**

### Property 10: Ad Format Character Limits

*For any* generated ad content, the character count must not exceed the format limit: SHORT ≤ 50 characters, MEDIUM ≤ 150 characters, LONG ≤ 300 characters.

**Validates: Requirements 4.2**

### Property 11: Ad Use Case Assignment

*For any* generated ad content, it must have a use_case field set to one of the valid e-wallet use cases: "cashback", "merchant_promo", or "payment_convenience".

**Validates: Requirements 4.3**

### Property 12: Minimum Ad Variations

*For any* segment and ad format combination, the Ad Generator must create at least 3 distinct ad content variations.

**Validates: Requirements 4.4**

### Property 13: Campaign Segment Association

*For any* campaign with target segments, the campaign must allow selection of one or more segments, and after activation, associations must exist between the campaign and all selected segments.

**Validates: Requirements 5.1, 5.3**

### Property 14: Reach Calculation Accuracy

*For any* set of selected segments, the estimated reach must equal the sum of unique customer counts across all selected segments.

**Validates: Requirements 5.2**

### Property 15: Minimum Segment Size Enforcement

*For any* campaign creation attempt, if any target segment has fewer than 100 customers, the campaign creation must be rejected with an appropriate error.

**Validates: Requirements 5.5**

### Property 16: Assignment Explanation Completeness

*For any* customer-segment assignment, there must be a non-empty LLM-generated natural language explanation and exactly 3 contributing features (identified from PCA loadings), each with an importance score in the range [0.0, 1.0].

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 17: Explanation Data References

*For any* customer-segment assignment explanation, the explanation text must contain references to at least one field from the customer profile (age, location, transaction_frequency, average_transaction_value, or merchant_categories).

**Validates: Requirements 6.3**

### Property 18: Explanation Regeneration on Segment Changes

*For any* segment refinement that changes segment criteria, all customer assignments to that segment must have their explanations regenerated with updated timestamps.

**Validates: Requirements 6.5**

### Property 19: Segment Distribution Consistency

*For any* set of segments, the sum of customer counts across all segments must equal the total number of customers, and the sum of segment percentages must equal 100% (within floating-point tolerance).

**Validates: Requirements 7.1**

### Property 20: Campaign Metrics Calculation

*For any* campaign with performance data, the click-through rate must equal clicks divided by impressions, and the conversion rate must equal conversions divided by clicks (or 0 if clicks is 0).

**Validates: Requirements 7.2**

> **Properties 21-25 deferred to production:** Report Export Round-Trip (Req 7.5), PII Anonymization (Req 8.1), Encryption Round-Trip (Req 8.2), Access Control Enforcement (Req 8.3), Audit Log Completeness (Req 8.5).

### Property 26: LLM Parameter Configuration

*For any* LLM configuration, all parameters (temperature, max_tokens, top_p, frequency_penalty, presence_penalty) must be settable and retrievable with the same values.

**Validates: Requirements 9.3**

### Property 27: LLM Retry Logic

*For any* LLM API call that fails, the system must retry exactly 3 times with exponentially increasing delays before returning an error, and each retry attempt must be logged.

**Validates: Requirements 9.4**

### Property 28: Segment Refinement Triggers Re-assignment

*For any* re-clustering operation with a new k value, all customers must be re-evaluated and assigned to new clusters based on the updated K-Means model.

**Validates: Requirements 10.3**

### Property 29: Segment Version History

*For any* re-clustering operation, a new version entry must be created with the previous clustering parameters (k value, centroids), timestamp, and change description, allowing rollback to any previous clustering configuration.

**Validates: Requirements 10.4, 10.5**

### Property 30: Chatbot Response Completeness

*For any* chatbot query that can be answered with available data, the response must contain both a non-empty natural language text answer and supporting data (if applicable).

**Validates: Requirements 11.4**

### Property 31: Conversation Context Persistence

*For any* chatbot session, all queries and responses within that session must be stored in conversation history and retrievable for context-aware follow-up questions.

**Validates: Requirements 11.5**

### Property 32: Unanswerable Query Handling

*For any* chatbot query that cannot be answered with available data, the response must contain an explanation of the limitation and at least one suggested alternative question.

**Validates: Requirements 11.6**



## Error Handling

### Error Categories

The system handles errors across multiple categories:

1. **Data Validation Errors**: Invalid or incomplete customer data
2. **LLM Provider Errors**: API failures, rate limits, timeouts
3. **Business Logic Errors**: Constraint violations (e.g., segment size < 100)
4. **System Errors**: Database failures, network issues

### Error Handling Strategies

#### 1. Data Validation Errors

**Strategy**: Fail gracefully, log errors, continue processing valid records

```python
class DataValidationError(Exception):
    def __init__(self, record_id: str, field: str, reason: str):
        self.record_id = record_id
        self.field = field
        self.reason = reason
        super().__init__(f"Validation failed for {record_id}.{field}: {reason}")
```

**Handling**:
- Log validation error with record identifier and reason
- Skip invalid record and continue processing batch
- Return summary with counts of successful and failed records
- Provide detailed error report for failed records

#### 2. LLM Provider Errors

**Strategy**: Retry with exponential backoff, fallback to alternative provider if configured

```python
class LLMProviderError(Exception):
    def __init__(self, provider: str, error_code: str, message: str):
        self.provider = provider
        self.error_code = error_code
        super().__init__(f"LLM Provider {provider} error [{error_code}]: {message}")
```

**Handling**:
- Implement exponential backoff: retry after 1s, 2s, 4s
- Log each retry attempt with timestamp
- After 3 failed retries, check for fallback provider
- If no fallback, return error to caller with detailed context
- For rate limit errors (429), use longer backoff periods

**Retry Logic**:
```python
def call_llm_with_retry(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return llm_provider.generate(prompt)
        except LLMProviderError as e:
            if attempt < max_retries - 1:
                delay = 2 ** attempt  # Exponential backoff
                logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts")
                raise
```

#### 3. Business Logic Errors

**Strategy**: Validate constraints early, return clear error messages

```python
class BusinessLogicError(Exception):
    def __init__(self, constraint: str, message: str):
        self.constraint = constraint
        super().__init__(f"Business constraint violated [{constraint}]: {message}")
```

**Handling**:
- Validate business rules before executing operations
- Return HTTP 400 Bad Request with detailed error message
- Examples:
  - Campaign targeting segment with < 100 customers
  - Duplicate segment names
  - Invalid confidence score ranges

#### 4. System Errors

**Strategy**: Log errors, return 500 Internal Server Error, alert operations team

```python
class SystemError(Exception):
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(f"System error in {component}: {message}")
```

**Handling**:
- Log full stack trace and context
- Return HTTP 500 with generic message
- Trigger alerting for critical errors (database down, disk full)
- Implement circuit breaker pattern for external dependencies

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Customer data validation failed",
    "details": {
      "failed_records": 5,
      "errors": [
        {
          "record_id": "cust_123",
          "field": "age",
          "reason": "Age must be between 0 and 120"
        }
      ]
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Graceful Degradation

When LLM services are unavailable:
- Segmentation: Use rule-based fallback (e.g., RFM segmentation)
- Ad Generation: Use template-based generation with segment variables
- Chatbot: Return message indicating service is temporarily unavailable
- Explainability: Return basic statistical explanations without LLM enhancement


## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property-based tests**: Verify universal properties across all inputs

Both testing approaches are complementary and necessary. Unit tests focus on concrete scenarios and integration points, while property-based tests ensure correctness across a wide range of inputs through randomization.

### Unit Testing

**Focus Areas**:
- Specific examples demonstrating correct behavior
- Integration points between components
- Edge cases and error conditions
- LLM provider adapter implementations

**Example Unit Tests**:

```python
def test_parse_valid_json_customer_data():
    """Verify JSON parsing with valid customer data"""
    json_data = '{"customer_id": "c1", "age": 25, "location": "Manila"}'
    result = segmentation_service.ingest_customer_data([json_data], DataFormat.JSON)
    assert result.success_count == 1
    assert result.failed_count == 0

def test_reject_campaign_with_small_segment():
    """Verify campaigns cannot target segments with < 100 customers"""
    small_segment = create_segment_with_customers(50)
    with pytest.raises(BusinessLogicError) as exc:
        targeting_engine.create_campaign("Test Campaign", [small_segment.id], ad_content)
    assert "minimum 100 customers" in str(exc.value)

def test_openai_adapter_validates_credentials():
    """Verify OpenAI adapter validates API credentials"""
    adapter = OpenAIAdapter(api_key="invalid_key")
    assert adapter.validate_credentials() == False
```

### Property-Based Testing

**Property-Based Testing Library**: Use **Hypothesis** (Python) or **fast-check** (JavaScript/TypeScript)

**Configuration**: Each property test must run a minimum of 100 iterations to ensure comprehensive input coverage.

**Test Tagging**: Each property test must include a comment tag referencing the design document property:

```python
# Feature: llm-customer-segmentation-ads, Property 1: Data Ingestion Round-Trip
@given(customer_profiles=st.lists(st.builds(CustomerProfile)))
@settings(max_examples=100)
def test_data_ingestion_round_trip(customer_profiles):
    """Property: Parsing and serializing customer data preserves all fields"""
    ...
```

**Property Test Examples**:

```python
# Feature: llm-customer-segmentation-ads, Property 4: Segment Count Bounds
@given(customers=st.lists(st.builds(CustomerProfile), min_size=100, max_size=10000))
@settings(max_examples=100)
def test_segment_count_bounds(customers):
    """Property: Number of segments created is between 3 and 10"""
    segments = segmentation_service.create_segments(customers)
    assert 3 <= len(segments) <= 10

# Feature: llm-customer-segmentation-ads, Property 6: Unique Segment Assignment
@given(
    customers=st.lists(st.builds(CustomerProfile), min_size=10),
    segments=st.lists(st.builds(Segment), min_size=3, max_size=10)
)
@settings(max_examples=100)
def test_unique_segment_assignment(customers, segments):
    """Property: Each customer assigned to exactly one segment"""
    assignments = segmentation_service.assign_customers_to_segments(
        [c.customer_id for c in customers], 
        segments
    )
    
    # Each customer appears exactly once
    customer_ids = [a.customer_id for a in assignments]
    assert len(customer_ids) == len(set(customer_ids))
    assert len(customer_ids) == len(customers)

# Feature: llm-customer-segmentation-ads, Property 7: Confidence Score Bounds
@given(assignments=st.lists(st.builds(CustomerSegmentAssignment)))
@settings(max_examples=100)
def test_confidence_score_bounds(assignments):
    """Property: All confidence scores are in range [0.0, 1.0]"""
    for assignment in assignments:
        assert 0.0 <= assignment.confidence_score <= 1.0

# Feature: llm-customer-segmentation-ads, Property 10: Ad Format Character Limits
@given(
    segment=st.builds(Segment),
    format=st.sampled_from([AdFormat.SHORT, AdFormat.MEDIUM, AdFormat.LONG])
)
@settings(max_examples=100)
def test_ad_format_character_limits(segment, format):
    """Property: Generated ads respect character limits"""
    ads = ad_generator.create_campaign_ads(segment.id, "Test campaign", [format])
    
    limits = {
        AdFormat.SHORT: 50,
        AdFormat.MEDIUM: 150,
        AdFormat.LONG: 300
    }
    
    for ad in ads.ads:
        assert len(ad.content) <= limits[ad.format]

# Feature: llm-customer-segmentation-ads, Property 14: Reach Calculation Accuracy
@given(segments=st.lists(st.builds(Segment), min_size=1, max_size=5))
@settings(max_examples=100)
def test_reach_calculation_accuracy(segments):
    """Property: Estimated reach equals sum of unique customers in segments"""
    segment_ids = [s.segment_id for s in segments]
    reach = targeting_engine.calculate_reach(segment_ids)
    
    expected_reach = sum(s.size for s in segments)
    assert reach.total_customers == expected_reach

# Feature: llm-customer-segmentation-ads, Property 19: Segment Distribution Consistency
@given(segments=st.lists(st.builds(Segment), min_size=3, max_size=10))
@settings(max_examples=100)
def test_segment_distribution_consistency(segments):
    """Property: Segment percentages sum to 100%"""
    distribution = analytics_service.get_segment_distribution(segments)
    
    total_customers = sum(s.size for s in segments)
    calculated_total = sum(d.customer_count for d in distribution)
    assert calculated_total == total_customers
    
    percentage_sum = sum(d.percentage for d in distribution)
    assert abs(percentage_sum - 100.0) < 0.01  # Floating-point tolerance

# Feature: llm-customer-segmentation-ads, Property 27: LLM Retry Logic
@given(prompt=st.text(min_size=10))
@settings(max_examples=100)
def test_llm_retry_logic(prompt, mocker):
    """Property: Failed LLM calls retry exactly 3 times"""
    mock_provider = mocker.Mock()
    mock_provider.generate.side_effect = LLMProviderError("test", "500", "Server error")
    
    llm_engine.provider = mock_provider
    
    with pytest.raises(LLMProviderError):
        llm_engine.call_llm(prompt, LLMProvider.OPENAI, LLMParameters())
    
    assert mock_provider.generate.call_count == 3

# Feature: llm-customer-segmentation-ads, Property 31: Conversation Context Persistence
@given(
    session_id=st.text(min_size=5),
    queries=st.lists(st.text(min_size=10), min_size=1, max_size=10)
)
@settings(max_examples=100)
def test_conversation_context_persistence(session_id, queries):
    """Property: All queries in session are stored in conversation history"""
    for query in queries:
        chatbot_service.process_query(query, session_id)
    
    context = chatbot_service.get_conversation_context(session_id)
    
    # All queries should be in history
    user_messages = [msg for msg in context.conversation_history if msg.role == MessageRole.USER]
    assert len(user_messages) == len(queries)
    
    # Queries should be in order
    for i, query in enumerate(queries):
        assert user_messages[i].content == query
```

### Integration Testing

**Focus Areas**:
- End-to-end workflows (data ingestion → segmentation → ad generation → campaign activation)
- LLM provider integrations with real API calls (using test accounts)
- Database operations and transactions
- Chatbot query processing with real LLM responses

**Example Integration Tests**:

```python
def test_complete_segmentation_workflow():
    """Test full workflow from data ingestion to segment creation"""
    # Ingest customer data
    customers = load_test_customer_data()
    result = segmentation_service.ingest_customer_data(customers, DataFormat.JSON)
    assert result.success_count > 0
    
    # Create segments
    segments = segmentation_service.create_segments([c.customer_id for c in customers])
    assert 3 <= len(segments) <= 10
    
    # Verify assignments
    for customer in customers:
        assignment = segmentation_service.get_customer_assignment(customer.customer_id)
        assert assignment is not None
        assert assignment.segment_id in [s.segment_id for s in segments]

def test_campaign_creation_and_activation():
    """Test campaign creation with ad generation and activation"""
    # Create segment
    segment = create_test_segment_with_customers(150)
    
    # Generate ads
    ads = ad_generator.create_campaign_ads(segment.id, "Test Campaign", [AdFormat.SHORT])
    assert len(ads.ads) >= 3
    
    # Create and activate campaign
    campaign = targeting_engine.create_campaign("Test Campaign", [segment.id], ads)
    result = targeting_engine.activate_campaign(campaign.campaign_id)
    assert result.status == "success"
    
    # Verify associations
    campaign_data = campaign_repository.get_campaign(campaign.campaign_id)
    assert campaign_data.status == CampaignStatus.ACTIVE
```

### Performance Testing

**Focus Areas**:
- Data ingestion performance (10,000 records in < 60 seconds)
- Segment profile query response time (< 2 seconds)
- Chatbot query response time (< 3 seconds simple, < 10 seconds complex)
- Report generation and export performance

**Tools**: Use **Locust** or **k6** for load testing

### Security Testing

**Focus Areas**:
- PII anonymization verification
- Encryption strength validation
- Access control enforcement
- SQL injection and XSS prevention
- API authentication and authorization

**Tools**: Use **OWASP ZAP** or **Burp Suite** for security scanning

### Test Coverage Goals

- **Unit Test Coverage**: Minimum 80% code coverage
- **Property Test Coverage**: All 32 correctness properties must have corresponding property tests
- **Integration Test Coverage**: All major workflows and component interactions
- **Security Test Coverage**: All authentication, authorization, and data protection mechanisms

### Continuous Integration

All tests should run automatically on:
- Every pull request
- Every commit to main branch
- Nightly builds (including longer-running performance tests)

**CI Pipeline**:
1. Run unit tests (fast feedback)
2. Run property-based tests (100 iterations each)
3. Run integration tests
4. Run security scans
5. Generate coverage reports
6. Block merge if tests fail or coverage drops below threshold



## Synthetic Data Generation Plan

### Overview

To effectively test the POC system, we need realistic synthetic customer data that exhibits natural clustering patterns suitable for PCA and K-Means segmentation. The synthetic data should represent diverse e-wallet user behaviors across multiple customer segments.

### Data Generation Goals

1. **Realistic Distributions**: Generate data that mimics real-world e-wallet usage patterns
2. **Natural Clustering**: Create distinct customer segments with separable characteristics
3. **Sufficient Volume**: Generate 10,000+ customer records for performance testing
4. **Feature Diversity**: Include all required customer attributes with realistic correlations
5. **Edge Cases**: Include outliers, missing data scenarios, and boundary conditions

### Customer Segment Archetypes

Design synthetic data around 5-7 distinct customer archetypes that should naturally cluster:

#### Archetype 1: High-Value Frequent Users
- **Age**: 30-45 years
- **Transaction Frequency**: 50-100 transactions/month
- **Average Transaction Value**: PHP 2,000-5,000
- **Top Categories**: Shopping, Dining, Entertainment
- **Payment Methods**: Credit card, e-wallet balance
- **Location**: Metro Manila, Makati, BGC

#### Archetype 2: Budget-Conscious Shoppers
- **Age**: 25-35 years
- **Transaction Frequency**: 30-50 transactions/month
- **Average Transaction Value**: PHP 300-800
- **Top Categories**: Groceries, Transportation, Utilities
- **Payment Methods**: Debit card, bank transfer
- **Location**: Quezon City, Pasig, Mandaluyong

#### Archetype 3: Bill Payment Users
- **Age**: 35-55 years
- **Transaction Frequency**: 10-20 transactions/month
- **Average Transaction Value**: PHP 1,500-3,000
- **Top Categories**: Utilities, Telecom, Insurance
- **Payment Methods**: Bank transfer, e-wallet balance
- **Location**: Various suburban areas

#### Archetype 4: Young Digital Natives
- **Age**: 18-25 years
- **Transaction Frequency**: 40-70 transactions/month
- **Average Transaction Value**: PHP 200-600
- **Top Categories**: Food delivery, Gaming, Streaming services
- **Payment Methods**: E-wallet balance, prepaid cards
- **Location**: University areas, Diliman, Taft

#### Archetype 5: Occasional Users
- **Age**: 40-60 years
- **Transaction Frequency**: 5-15 transactions/month
- **Average Transaction Value**: PHP 500-1,500
- **Top Categories**: Groceries, Healthcare, Transportation
- **Payment Methods**: Debit card, cash-in
- **Location**: Provincial cities

#### Archetype 6: Business/Merchant Users
- **Age**: 30-50 years
- **Transaction Frequency**: 80-150 transactions/month
- **Average Transaction Value**: PHP 5,000-15,000
- **Top Categories**: Wholesale, Business services, Supplies
- **Payment Methods**: Bank transfer, credit card
- **Location**: Business districts

#### Archetype 7: Remittance Users
- **Age**: 25-45 years
- **Transaction Frequency**: 15-30 transactions/month
- **Average Transaction Value**: PHP 3,000-8,000
- **Top Categories**: Money transfer, Cash-out, International remittance
- **Payment Methods**: Bank transfer, remittance services
- **Location**: OFW-heavy areas

### Data Generation Implementation

#### Python Implementation Using Faker and NumPy

```python
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import json
import hashlib

fake = Faker(['en_PH', 'en_US'])  # Philippine locale

class SyntheticDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = fake
        
        # Define merchant categories
        self.merchant_categories = {
            'Shopping': ['Fashion', 'Electronics', 'Home & Living', 'Beauty'],
            'Dining': ['Restaurants', 'Fast Food', 'Cafes', 'Food Delivery'],
            'Entertainment': ['Movies', 'Gaming', 'Streaming', 'Events'],
            'Groceries': ['Supermarket', 'Convenience Store', 'Wet Market'],
            'Transportation': ['Ride-hailing', 'Public Transport', 'Parking', 'Fuel'],
            'Utilities': ['Electricity', 'Water', 'Internet', 'Phone'],
            'Healthcare': ['Pharmacy', 'Hospital', 'Clinic', 'Insurance'],
            'Business': ['Wholesale', 'Office Supplies', 'Professional Services'],
            'Remittance': ['Money Transfer', 'Cash-out', 'International Transfer']
        }
        
        # Define locations (Philippine cities)
        self.locations = [
            'Makati', 'BGC', 'Quezon City', 'Pasig', 'Mandaluyong',
            'Manila', 'Taguig', 'Paranaque', 'Las Pinas', 'Muntinlupa',
            'Caloocan', 'Valenzuela', 'Malabon', 'Navotas',
            'Cebu City', 'Davao City', 'Iloilo City', 'Bacolod', 'Cagayan de Oro'
        ]
        
        self.payment_methods = [
            'Credit Card', 'Debit Card', 'E-wallet Balance', 
            'Bank Transfer', 'Prepaid Card', 'Cash-in'
        ]
    
    def generate_customer_profile(self, archetype: dict) -> dict:
        """Generate a single customer profile based on archetype"""
        
        # Generate base attributes with some randomness
        age = np.random.randint(archetype['age_range'][0], archetype['age_range'][1])
        transaction_frequency = np.random.randint(
            archetype['transaction_freq'][0], 
            archetype['transaction_freq'][1]
        )
        avg_transaction_value = np.random.uniform(
            archetype['avg_transaction'][0],
            archetype['avg_transaction'][1]
        )
        
        # Generate customer ID (anonymized hash)
        raw_id = f"{self.fake.uuid4()}"
        customer_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        
        # Select location with bias towards archetype preferences
        location = np.random.choice(
            archetype['locations'],
            p=archetype.get('location_weights', None)
        )
        
        # Select merchant categories
        merchant_categories = np.random.choice(
            archetype['top_categories'],
            size=min(3, len(archetype['top_categories'])),
            replace=False
        ).tolist()
        
        # Select payment methods
        preferred_payment_methods = np.random.choice(
            archetype['payment_methods'],
            size=min(2, len(archetype['payment_methods'])),
            replace=False
        ).tolist()
        
        # Calculate derived attributes
        total_spend = avg_transaction_value * transaction_frequency * np.random.uniform(0.8, 1.2)
        account_age_days = np.random.randint(30, 1095)  # 1 month to 3 years
        
        # Generate last transaction date (within last 30 days)
        last_transaction_date = datetime.now() - timedelta(days=np.random.randint(0, 30))
        
        return {
            'customer_id': customer_id,
            'age': int(age),
            'location': location,
            'transaction_frequency': int(transaction_frequency),
            'average_transaction_value': round(avg_transaction_value, 2),
            'merchant_categories': merchant_categories,
            'total_spend': round(total_spend, 2),
            'account_age_days': int(account_age_days),
            'preferred_payment_methods': preferred_payment_methods,
            'last_transaction_date': last_transaction_date.isoformat(),
            'created_at': (datetime.now() - timedelta(days=account_age_days)).isoformat(),
            'updated_at': datetime.now().isoformat()
        }
    
    def generate_dataset(self, total_customers=10000, archetype_distribution=None):
        """Generate complete synthetic dataset"""
        
        # Define archetypes
        archetypes = {
            'high_value_frequent': {
                'age_range': (30, 45),
                'transaction_freq': (50, 100),
                'avg_transaction': (2000, 5000),
                'top_categories': ['Shopping', 'Dining', 'Entertainment'],
                'payment_methods': ['Credit Card', 'E-wallet Balance'],
                'locations': ['Makati', 'BGC', 'Taguig'],
                'location_weights': [0.4, 0.4, 0.2]
            },
            'budget_conscious': {
                'age_range': (25, 35),
                'transaction_freq': (30, 50),
                'avg_transaction': (300, 800),
                'top_categories': ['Groceries', 'Transportation', 'Utilities'],
                'payment_methods': ['Debit Card', 'Bank Transfer'],
                'locations': ['Quezon City', 'Pasig', 'Mandaluyong'],
                'location_weights': [0.5, 0.3, 0.2]
            },
            'bill_payment': {
                'age_range': (35, 55),
                'transaction_freq': (10, 20),
                'avg_transaction': (1500, 3000),
                'top_categories': ['Utilities', 'Healthcare', 'Insurance'],
                'payment_methods': ['Bank Transfer', 'E-wallet Balance'],
                'locations': ['Manila', 'Paranaque', 'Las Pinas', 'Muntinlupa'],
                'location_weights': [0.3, 0.25, 0.25, 0.2]
            },
            'young_digital': {
                'age_range': (18, 25),
                'transaction_freq': (40, 70),
                'avg_transaction': (200, 600),
                'top_categories': ['Dining', 'Entertainment', 'Gaming'],
                'payment_methods': ['E-wallet Balance', 'Prepaid Card'],
                'locations': ['Quezon City', 'Manila', 'Pasig'],
                'location_weights': [0.5, 0.3, 0.2]
            },
            'occasional': {
                'age_range': (40, 60),
                'transaction_freq': (5, 15),
                'avg_transaction': (500, 1500),
                'top_categories': ['Groceries', 'Healthcare', 'Transportation'],
                'payment_methods': ['Debit Card', 'Cash-in'],
                'locations': ['Cebu City', 'Davao City', 'Iloilo City', 'Bacolod'],
                'location_weights': [0.4, 0.3, 0.2, 0.1]
            },
            'business_merchant': {
                'age_range': (30, 50),
                'transaction_freq': (80, 150),
                'avg_transaction': (5000, 15000),
                'top_categories': ['Business', 'Shopping', 'Wholesale'],
                'payment_methods': ['Bank Transfer', 'Credit Card'],
                'locations': ['Makati', 'BGC', 'Quezon City'],
                'location_weights': [0.5, 0.3, 0.2]
            },
            'remittance': {
                'age_range': (25, 45),
                'transaction_freq': (15, 30),
                'avg_transaction': (3000, 8000),
                'top_categories': ['Remittance', 'Cash-out', 'Money Transfer'],
                'payment_methods': ['Bank Transfer', 'Remittance Services'],
                'locations': ['Manila', 'Quezon City', 'Caloocan', 'Valenzuela'],
                'location_weights': [0.3, 0.3, 0.2, 0.2]
            }
        }
        
        # Default distribution if not provided
        if archetype_distribution is None:
            archetype_distribution = {
                'high_value_frequent': 0.15,
                'budget_conscious': 0.25,
                'bill_payment': 0.15,
                'young_digital': 0.20,
                'occasional': 0.10,
                'business_merchant': 0.08,
                'remittance': 0.07
            }
        
        # Generate customers
        customers = []
        for archetype_name, proportion in archetype_distribution.items():
            num_customers = int(total_customers * proportion)
            archetype = archetypes[archetype_name]
            
            for _ in range(num_customers):
                customer = self.generate_customer_profile(archetype)
                customer['archetype'] = archetype_name  # For validation only
                customers.append(customer)
        
        # Add some noise/outliers (5% of dataset)
        num_outliers = int(total_customers * 0.05)
        for _ in range(num_outliers):
            # Random archetype
            archetype = archetypes[np.random.choice(list(archetypes.keys()))]
            customer = self.generate_customer_profile(archetype)
            
            # Add noise to make it an outlier
            customer['transaction_frequency'] = int(customer['transaction_frequency'] * np.random.uniform(0.3, 2.5))
            customer['average_transaction_value'] = customer['average_transaction_value'] * np.random.uniform(0.2, 3.0)
            customer['archetype'] = 'outlier'
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_transaction_history(self, customer_profile: dict, num_transactions: int = None):
        """Generate transaction history for a customer"""
        
        if num_transactions is None:
            num_transactions = customer_profile['transaction_frequency']
        
        transactions = []
        base_date = datetime.fromisoformat(customer_profile['last_transaction_date'])
        
        for i in range(num_transactions):
            # Generate transaction date (spread over last 30 days)
            days_ago = np.random.randint(0, 30)
            transaction_date = base_date - timedelta(days=days_ago)
            
            # Select merchant category
            merchant_category = np.random.choice(customer_profile['merchant_categories'])
            
            # Generate transaction amount (around average with variance)
            amount = customer_profile['average_transaction_value'] * np.random.lognormal(0, 0.5)
            amount = max(50, min(amount, customer_profile['average_transaction_value'] * 3))
            
            # Select payment method
            payment_method = np.random.choice(customer_profile['preferred_payment_methods'])
            
            transaction = {
                'transaction_id': self.fake.uuid4(),
                'customer_id': customer_profile['customer_id'],
                'amount': round(amount, 2),
                'merchant_category': merchant_category,
                'merchant_name': self.fake.company(),
                'transaction_type': np.random.choice(['payment', 'transfer', 'cashout'], p=[0.7, 0.2, 0.1]),
                'timestamp': transaction_date.isoformat(),
                'payment_method': payment_method,
                'location': customer_profile['location']
            }
            transactions.append(transaction)
        
        return transactions
    
    def export_to_json(self, df: pd.DataFrame, filename: str):
        """Export dataset to JSON format"""
        # Remove archetype column (used for validation only)
        df_export = df.drop(columns=['archetype'], errors='ignore')
        df_export.to_json(filename, orient='records', indent=2)
        print(f"Exported {len(df_export)} customer profiles to {filename}")
    
    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """Export dataset to CSV format"""
        # Remove archetype column
        df_export = df.drop(columns=['archetype'], errors='ignore')
        
        # Flatten list columns for CSV
        df_export['merchant_categories'] = df_export['merchant_categories'].apply(lambda x: '|'.join(x))
        df_export['preferred_payment_methods'] = df_export['preferred_payment_methods'].apply(lambda x: '|'.join(x))
        
        df_export.to_csv(filename, index=False)
        print(f"Exported {len(df_export)} customer profiles to {filename}")
    
    def generate_validation_report(self, df: pd.DataFrame):
        """Generate report showing archetype distribution and statistics"""
        print("\n=== Synthetic Data Generation Report ===\n")
        print(f"Total Customers: {len(df)}")
        print(f"\nArchetype Distribution:")
        print(df['archetype'].value_counts())
        
        print(f"\n=== Feature Statistics ===")
        print(f"\nAge Distribution:")
        print(df['age'].describe())
        
        print(f"\nTransaction Frequency Distribution:")
        print(df['transaction_frequency'].describe())
        
        print(f"\nAverage Transaction Value Distribution:")
        print(df['average_transaction_value'].describe())
        
        print(f"\nLocation Distribution (Top 10):")
        print(df['location'].value_counts().head(10))
        
        print(f"\n=== Clustering Suitability Check ===")
        # Check if data has sufficient variance for PCA
        numeric_features = ['age', 'transaction_frequency', 'average_transaction_value', 'total_spend', 'account_age_days']
        feature_matrix = df[numeric_features].values
        
        print(f"Feature variance:")
        for i, feature in enumerate(numeric_features):
            print(f"  {feature}: {np.var(feature_matrix[:, i]):.2f}")
        
        # Check correlation between features
        print(f"\nFeature correlations:")
        correlation_matrix = np.corrcoef(feature_matrix.T)
        for i, feature1 in enumerate(numeric_features):
            for j, feature2 in enumerate(numeric_features):
                if i < j:
                    print(f"  {feature1} <-> {feature2}: {correlation_matrix[i, j]:.3f}")


# Usage Example
if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate 10,000 customer profiles
    print("Generating synthetic customer data...")
    df = generator.generate_dataset(total_customers=10000)
    
    # Generate validation report
    generator.generate_validation_report(df)
    
    # Export to JSON and CSV
    generator.export_to_json(df, 'synthetic_customers.json')
    generator.export_to_csv(df, 'synthetic_customers.csv')
    
    # Generate transaction history for first 100 customers (sample)
    print("\nGenerating transaction history for sample customers...")
    all_transactions = []
    for idx, customer in df.head(100).iterrows():
        transactions = generator.generate_transaction_history(customer.to_dict())
        all_transactions.extend(transactions)
    
    transactions_df = pd.DataFrame(all_transactions)
    transactions_df.to_json('synthetic_transactions_sample.json', orient='records', indent=2)
    print(f"Exported {len(transactions_df)} transactions to synthetic_transactions_sample.json")
```

### Data Validation and Quality Checks

After generating synthetic data, perform these validation checks:

#### 1. Distribution Validation
```python
def validate_distributions(df: pd.DataFrame):
    """Validate that generated data follows expected distributions"""
    
    # Check age distribution
    assert df['age'].min() >= 18, "Age should be >= 18"
    assert df['age'].max() <= 60, "Age should be <= 60"
    
    # Check transaction frequency
    assert df['transaction_frequency'].min() >= 5, "Min transaction frequency should be >= 5"
    assert df['transaction_frequency'].max() <= 150, "Max transaction frequency should be <= 150"
    
    # Check for required fields
    required_fields = ['customer_id', 'age', 'location', 'transaction_frequency', 
                       'average_transaction_value', 'merchant_categories']
    for field in required_fields:
        assert field in df.columns, f"Missing required field: {field}"
        assert df[field].notna().all(), f"Field {field} contains null values"
    
    print("✓ Distribution validation passed")
```

#### 2. Clustering Suitability Check
```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def check_clustering_suitability(df: pd.DataFrame):
    """Check if synthetic data is suitable for PCA and K-Means"""
    
    # Prepare features
    numeric_features = ['age', 'transaction_frequency', 'average_transaction_value', 
                        'total_spend', 'account_age_days']
    X = df[numeric_features].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check PCA variance
    pca = PCA()
    pca.fit(X_scaled)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
    
    print(f"✓ PCA: {n_components_80} components explain 80% variance")
    print(f"  Explained variance by component: {pca.explained_variance_ratio_[:5]}")
    
    # Check K-Means clustering
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    for k in range(3, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append((k, score))
    
    print(f"\n✓ K-Means silhouette scores:")
    for k, score in silhouette_scores:
        print(f"  k={k}: {score:.3f}")
    
    best_k = max(silhouette_scores, key=lambda x: x[1])
    print(f"\n✓ Optimal k: {best_k[0]} (silhouette score: {best_k[1]:.3f})")
```

### Test Data Scenarios

Generate specialized datasets for different test scenarios:

#### Scenario 1: Performance Testing (Large Dataset)
```python
# Generate 50,000 customers for performance testing
large_dataset = generator.generate_dataset(total_customers=50000)
generator.export_to_json(large_dataset, 'synthetic_customers_large.json')
```

#### Scenario 2: Edge Cases
```python
# Generate dataset with edge cases
edge_cases = [
    # Very young user
    {'age': 18, 'transaction_frequency': 5, 'avg_transaction': 100},
    # Very old user
    {'age': 60, 'transaction_frequency': 5, 'avg_transaction': 500},
    # Extremely high frequency
    {'age': 35, 'transaction_frequency': 150, 'avg_transaction': 10000},
    # Very low activity
    {'age': 50, 'transaction_frequency': 1, 'avg_transaction': 50},
]
```

#### Scenario 3: Data Quality Issues
```python
# Generate dataset with intentional quality issues for testing error handling
def generate_problematic_data():
    """Generate data with quality issues"""
    df = generator.generate_dataset(total_customers=1000)
    
    # Introduce missing values (5%)
    df.loc[df.sample(frac=0.05).index, 'location'] = None
    
    # Introduce invalid ages (2%)
    df.loc[df.sample(frac=0.02).index, 'age'] = -1
    
    # Introduce duplicate customer IDs (1%)
    duplicates = df.sample(frac=0.01)
    df = pd.concat([df, duplicates])
    
    return df
```

### Data Generation Execution Plan

1. **Generate Base Dataset** (10,000 customers)
   - Run synthetic data generator with default archetype distribution
   - Validate distributions and clustering suitability
   - Export to JSON and CSV formats

2. **Generate Transaction History** (for 1,000 sample customers)
   - Generate detailed transaction records
   - Export for transaction analysis testing

3. **Generate Performance Test Dataset** (50,000 customers)
   - Test data ingestion performance
   - Test PCA and K-Means scalability

4. **Generate Edge Case Dataset** (500 customers)
   - Test boundary conditions
   - Test error handling

5. **Generate Data Quality Test Dataset** (1,000 customers with issues)
   - Test validation logic
   - Test error recovery

### Expected Outputs

After running the data generation plan:

```
synthetic_data/
├── synthetic_customers.json              # 10,000 customers (main dataset)
├── synthetic_customers.csv               # Same data in CSV format
├── synthetic_transactions_sample.json    # Transaction history for 1,000 customers
├── synthetic_customers_large.json        # 50,000 customers (performance testing)
├── synthetic_customers_edge_cases.json   # 500 edge case customers
├── synthetic_customers_quality_issues.json  # 1,000 customers with data issues
└── data_generation_report.txt            # Validation report
```

### Integration with Testing

Use generated synthetic data in tests:

```python
# In test files
def load_synthetic_data():
    """Load synthetic customer data for testing"""
    with open('synthetic_data/synthetic_customers.json', 'r') as f:
        return json.load(f)

def test_segmentation_with_synthetic_data():
    """Test segmentation using synthetic data"""
    customers = load_synthetic_data()
    
    # Ingest data
    result = segmentation_service.ingest_customer_data(customers, DataFormat.JSON)
    assert result.success_count == len(customers)
    
    # Create segments
    segments = segmentation_service.create_segments([c['customer_id'] for c in customers])
    assert 3 <= len(segments) <= 10
    
    # Verify clustering quality
    # ... additional assertions
```
