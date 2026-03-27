# LLM-Powered Customer Segmentation Advertising System

A proof-of-concept system that uses machine learning techniques (Principal Component Analysis and K-Means Clustering) combined with Large Language Models (LLMs) to deliver targeted advertising for e-wallet platforms.

## Features

- **ML-First Segmentation**: Uses PCA and K-Means clustering for data-driven customer segmentation
- **LLM Integration**: Leverages LLMs for natural language insights, ad content generation, and conversational analytics
- **Multi-Provider Support**: Supports OpenAI, Anthropic, and local LLM models
- **Privacy by Design**: Anonymizes PII, encrypts data at rest and in transit
- **Explainability**: Provides clear explanations for segmentation decisions
- **Interactive Chatbot**: Natural language query interface for data exploration

## Technology Stack

- **Backend**: Python 3.11+, FastAPI
- **ML Libraries**: scikit-learn, pandas, numpy
- **LLM Integration**: OpenAI SDK, Anthropic SDK
- **Testing**: pytest, hypothesis (property-based testing)
- **Package Management**: uv (Astral's package manager)

## Project Structure

```
.
├── src/
│   ├── config.py                    # Configuration management (env vars, providers)
│   ├── engines/                     # ML and LLM engines
│   │   ├── pca_engine.py            # PCA dimensionality reduction
│   │   ├── kmeans_engine.py         # K-Means clustering with optimal k
│   │   ├── llm_engine.py            # LLM orchestration with retry logic
│   │   └── adapters/                # LLM provider adapters
│   │       ├── base.py              # Abstract adapter interface
│   │       ├── openai_adapter.py    # OpenAI GPT integration
│   │       ├── anthropic_adapter.py # Anthropic Claude integration
│   │       └── local_adapter.py     # Local model (Ollama) integration
│   ├── services/                    # Application service layer
│   │   ├── segmentation_service.py  # PCA + K-Means + LLM pipeline
│   │   ├── ad_generator_service.py  # Personalized ad content generation
│   │   ├── targeting_engine.py      # Campaign targeting and reach estimation
│   │   ├── query_chatbot_service.py # Conversational analytics chatbot
│   │   └── synthetic_data_generator.py # Test data generation
│   ├── models/                      # Pydantic v2 data models
│   │   ├── customer.py              # CustomerProfile, TransactionData
│   │   ├── segment.py               # Segment, CustomerSegmentAssignment
│   │   ├── ml.py                    # PCAResult, ClusteringResult, ClusterStatistics
│   │   ├── campaign.py              # Campaign, AdContent, AdFormat
│   │   ├── llm.py                   # LLMConfiguration, LLMParameters
│   │   ├── chatbot.py               # ConversationContext, ChatMessage, QueryIntent
│   │   ├── audit.py                 # AuditLogEntry
│   │   └── service_models.py        # IngestionResult, ReachEstimate, etc.
│   ├── repositories/                # Data access layer (in-memory, extensible)
│   │   ├── customer_repository.py   # AES-256 encryption, PII anonymization
│   │   ├── segment_repository.py    # Versioned segment storage
│   │   └── campaign_repository.py   # Campaign and ad content storage
│   └── api/                         # FastAPI REST API endpoints (planned)
├── tests/                           # 174 tests (unit + property-based)
├── scripts/                         # Dataset generation and verification
├── synthetic_data/                  # Pre-generated test datasets
├── pyproject.toml                   # Project dependencies and configuration
└── README.md                        # This file
```

## Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Install uv (if not already installed):
```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via Homebrew
brew install uv
```

2. Clone the repository:
```bash
git clone <repository-url>
cd llm-customer-segmentation-ads
```

3. Install dependencies using uv (this automatically creates a virtual environment):
```bash
uv sync
```

4. Create a `.env` file from the example:
```bash
cp .env.example .env
```

5. Configure your LLM provider API keys in `.env`:
```bash
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Running the Application

Start the FastAPI development server:
```bash
uv run uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

API documentation (Swagger UI): `http://localhost:8000/docs`

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run tests with coverage:
```bash
uv run pytest --cov=src --cov-report=html
```

Run property-based tests only:
```bash
uv run pytest -m property
```

## Configuration

The application can be configured through environment variables or a `.env` file. See `.env.example` for all available configuration options.

### Key Configuration Options

- `DEFAULT_LLM_PROVIDER`: Choose between `openai`, `anthropic`, or `local`
- `PCA_VARIANCE_THRESHOLD`: Minimum variance explained by PCA (default: 0.8)
- `KMEANS_MIN_CLUSTERS` / `KMEANS_MAX_CLUSTERS`: Range for optimal cluster detection
- `MIN_SEGMENT_SIZE`: Minimum customers required in a segment for targeting (default: 100)

## Development

### Virtual Environment

uv automatically manages the virtual environment in `.venv/`. You don't need to manually activate it when using `uv run` commands.

To activate it manually (optional):
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

**Note:** When using `uv run`, the virtual environment is automatically activated for that command.

### Adding Dependencies

Add a new dependency:
```bash
uv add <package-name>
```

Add a development dependency:
```bash
uv add --dev <package-name>
```

Remove a dependency:
```bash
uv remove <package-name>
```

### uv Quick Reference

**Common Commands:**
- `uv sync` - Install/sync all dependencies from pyproject.toml
- `uv run <command>` - Run command in the virtual environment (auto-activates)
- `uv pip list` - List installed packages
- `uv pip freeze` - Show installed packages with versions
- `uv lock` - Update the lock file
- `uv python install 3.11` - Install a specific Python version

**Why uv?**
- 10-100x faster than pip
- Automatic virtual environment management
- Deterministic dependency resolution
- Built-in Python version management
- Compatible with pip and pyproject.toml standards

### Code Quality

Run linting and formatting:
```bash
uv run ruff check src/
uv run ruff format src/
```

## Architecture

The system follows a layered architecture:

```
Dashboard / API  -->  Service Layer  -->  Engine Layer  -->  Data Layer
```

**Engine Layer** (implemented):
- `PCAEngine` -- Dimensionality reduction (80% variance threshold)
- `KMeansEngine` -- Clustering with optimal k detection (silhouette score)
- `LLMEngine` -- Multi-provider LLM orchestration with retry logic

**Service Layer** (implemented):
- `SegmentationService` -- Orchestrates PCA + K-Means + LLM into a full segmentation pipeline (ingest, segment, assign, explain, refine)
- `AdGeneratorService` -- Generates personalized ad content with A/B variations and content validation
- `TargetingEngine` -- Campaign management with segment targeting, reach estimation, and minimum-size enforcement
- `QueryChatbotService` -- Conversational analytics with intent classification, session management, and context-aware responses

**Data Layer** (implemented):
- `CustomerDataRepository` -- AES-256 encrypted storage with PII anonymization
- `SegmentDataRepository` -- Version-tracked segment and assignment storage
- `CampaignDataRepository` -- Campaign, ad content, and metrics storage

## API Endpoints (Planned)

The REST API endpoints are defined in the design spec and will wrap the service layer:

### Data Ingestion
- `POST /api/v1/customers/ingest` - Ingest customer data (JSON/CSV)
- `GET /api/v1/customers/{customer_id}` - Retrieve customer profile

### Segmentation
- `POST /api/v1/segments/create` - Create segments from customer data
- `GET /api/v1/segments` - List all segments
- `GET /api/v1/segments/{segment_id}` - Get segment details
- `POST /api/v1/segments/refine` - Refine segments with new k value

### Ad Generation
- `POST /api/v1/ads/generate` - Generate ad content for segment
- `GET /api/v1/ads/{ad_id}` - Get ad content details

### Campaign Management
- `POST /api/v1/campaigns/create` - Create campaign
- `GET /api/v1/campaigns` - List campaigns
- `POST /api/v1/campaigns/{campaign_id}/activate` - Activate campaign

### Analytics
- `GET /api/v1/analytics/segments/distribution` - Get segment distribution
- `GET /api/v1/analytics/campaigns/{campaign_id}/performance` - Get campaign metrics

### Chatbot
- `POST /api/v1/chatbot/query` - Process chatbot query
- `GET /api/v1/chatbot/sessions/{session_id}/context` - Get conversation context

## Testing

174 tests covering engines, repositories, services, and data models.

```bash
uv run pytest              # Run all tests
uv run pytest -v           # Verbose output
uv run pytest --cov=src    # With coverage
```

The project uses a dual testing approach:

1. **Unit Tests** -- Verify specific examples, edge cases, and integration points
2. **Property-Based Tests** -- Verify universal properties across all inputs using Hypothesis (100 iterations per test)

### Test Suite Breakdown

| Module | Tests | Coverage |
|--------|-------|----------|
| Engines (PCA, K-Means, LLM) | 55 | Unit + property-based |
| Adapters (OpenAI, Anthropic, Local) | 9 | Unit |
| Repositories (Customer, Segment, Campaign) | 30 | Unit + property-based |
| Data Models | 18 | Validation + edge cases |
| Services (Segmentation, Ads, Targeting, Chatbot) | 49 | Unit |
| Setup & framework | 4 | Smoke tests |
| **Total** | **174** | |

## Usage Examples

### Run the Segmentation Pipeline

```python
from src.engines.pca_engine import PCAEngine
from src.engines.kmeans_engine import KMeansEngine
from src.engines.llm_engine import LLMEngine
from src.models.llm import LLMConfiguration, LLMParameters, LLMProvider
from src.repositories.customer_repository import CustomerDataRepository
from src.repositories.segment_repository import SegmentDataRepository
from src.services.segmentation_service import SegmentationService

# Initialize engines
pca = PCAEngine()
kmeans = KMeansEngine()
llm_config = LLMConfiguration(
    config_id="default",
    provider=LLMProvider.OPENAI,
    model_name="gpt-4",
    api_key="your-key",
    parameters=LLMParameters(),
)
llm = LLMEngine(llm_config)

# Initialize repositories
customer_repo = CustomerDataRepository()
segment_repo = SegmentDataRepository()

# Create service
service = SegmentationService(pca, kmeans, llm, customer_repo, segment_repo)

# Ingest data
result = service.ingest_customer_data(customer_records)

# Create segments (auto-determines optimal k)
segments = service.create_segments()

# Assign customers
customer_ids = [c.customer_id for c in customer_repo.list_customers()]
assignments = service.assign_customers_to_segments(customer_ids, segments)
```

### Generate Ad Content

```python
from src.services.ad_generator_service import AdGeneratorService
from src.models.campaign import AdFormat

ad_service = AdGeneratorService(llm, segment_repo, campaign_repo)
ads = ad_service.create_campaign_ads(
    segment_id="segment_0",
    campaign_id="summer_2025",
    campaign_brief="Cashback promo for frequent shoppers",
    formats=[AdFormat.SHORT, AdFormat.MEDIUM, AdFormat.LONG],
)
```

### Create and Activate a Campaign

```python
from src.services.targeting_engine import TargetingEngine

targeting = TargetingEngine(segment_repo, campaign_repo)
campaign = targeting.create_campaign(
    campaign_name="Summer Cashback",
    target_segment_ids=["segment_0", "segment_1"],
    ad_content_ids=[ad.ad_id for ad in ads],
)
result = targeting.activate_campaign(campaign.campaign_id)
```

### Chat with the Segmentation Data

```python
from src.services.query_chatbot_service import QueryChatbotService

chatbot = QueryChatbotService(llm, segment_repo, customer_repo, campaign_repo)
response = chatbot.process_query(
    "How many customers are in the Young Spenders segment?",
    session_id="session_001",
)
print(response.text)
```

## Implementation Status

| Phase | Status |
|-------|--------|
| 1. Project setup and dependencies | Done |
| 2. Data models and validation | Done |
| 3. Synthetic data generation | Done |
| 4. PCA Engine | Done |
| 5. K-Means Engine | Done |
| 6. LLM Engine and adapters | Done |
| 7. Checkpoint (engines) | Done |
| 8. Data repositories | Done |
| 9. Segmentation Service | Done |
| 10. Ad Generator Service | Done |
| 11. Targeting Engine | Done |
| 12. Query Chatbot Service | Done |
| 13. Checkpoint (services) | Done |
| 14. Security and access control | Planned |
| 15. FastAPI REST API endpoints | Planned |
| 16. Analytics and reporting | Planned |
| 17. Analytics Dashboard (frontend) | Planned |

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
