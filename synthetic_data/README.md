# Synthetic Data for Customer Segmentation Testing

This directory contains synthetic e-wallet customer data generated for testing the ML-powered customer segmentation advertising system.

## Generated Datasets

### 1. Main Dataset (10,000 customers)
- **Files**: `synthetic_customers.json`, `synthetic_customers.csv`
- **Purpose**: Primary dataset for development and testing
- **Size**: 10,500 customers (including 500 outliers)
- **Validation Report**: `main_dataset_report.txt`

**Archetype Distribution**:
- Budget-Conscious Shoppers: 2,500 (23.8%)
- Young Digital Natives: 2,000 (19.0%)
- High-Value Frequent Users: 1,500 (14.3%)
- Bill Payment Users: 1,500 (14.3%)
- Occasional Users: 1,000 (9.5%)
- Business/Merchant Users: 800 (7.6%)
- Remittance Users: 700 (6.7%)
- Outliers: 500 (4.8%)

**Clustering Suitability**:
- ✓ PCA: 3 components explain 80% variance
- ✓ Optimal K-Means clusters: 9 (silhouette score: 0.355)
- ✓ Features show good variance and correlation patterns

### 2. Large Dataset (50,000 customers)
- **Files**: `synthetic_customers_large.json`, `synthetic_customers_large.csv`
- **Purpose**: Performance testing and scalability validation
- **Size**: 52,500 customers (including outliers)
- **Use Cases**:
  - Test data ingestion performance (target: < 60 seconds)
  - Test PCA and K-Means scalability
  - Stress test clustering algorithms

### 3. Edge Case Dataset (500 customers)
- **Files**: `synthetic_customers_edge_cases.json`, `synthetic_customers_edge_cases.csv`
- **Purpose**: Test boundary conditions and edge cases
- **Size**: 500 customers (50 per edge case type)

**Edge Case Types**:
- Very young users (age 18-19)
- Very old users (age 59-60)
- Extremely high frequency users (145-150 transactions/month)
- Very low activity users (1-5 transactions/month)
- Extreme high transaction values (PHP 20,000-50,000)
- Extreme low transaction values (PHP 10-100)
- New accounts (1-30 days old)
- Old accounts (1000-1095 days old)
- Single category users (only 1 merchant category)
- Multi-category users (5+ merchant categories)

### 4. Data Quality Issues Dataset (1,000 customers)
- **Files**: `synthetic_customers_quality_issues.json`, `synthetic_customers_quality_issues.csv`
- **Purpose**: Test error handling and data validation
- **Size**: 1,060 customers (including duplicates)

**Intentional Issues**:
- Missing location values: 52 records (5%)
- Invalid ages (negative): 21 records (2%)
- Negative transaction frequencies: 21 records (2%)
- Negative transaction values: 21 records (2%)
- Duplicate customer IDs: 10 records (1%)
- Empty merchant categories: 11 records (1%)

### 5. Transaction History Sample
- **File**: `synthetic_transactions_sample.json`
- **Purpose**: Test transaction analysis and history processing
- **Size**: 75,022 transactions for 1,000 customers
- **Average**: ~75 transactions per customer

## Customer Archetypes

### 1. High-Value Frequent Users
- Age: 30-45 years
- Transaction Frequency: 50-100/month
- Avg Transaction: PHP 2,000-5,000
- Categories: Shopping, Dining, Entertainment
- Locations: Makati, BGC, Taguig

### 2. Budget-Conscious Shoppers
- Age: 25-35 years
- Transaction Frequency: 30-50/month
- Avg Transaction: PHP 300-800
- Categories: Groceries, Transportation, Utilities
- Locations: Quezon City, Pasig, Mandaluyong

### 3. Bill Payment Users
- Age: 35-55 years
- Transaction Frequency: 10-20/month
- Avg Transaction: PHP 1,500-3,000
- Categories: Utilities, Healthcare, Insurance
- Locations: Manila, Paranaque, Las Pinas, Muntinlupa

### 4. Young Digital Natives
- Age: 18-25 years
- Transaction Frequency: 40-70/month
- Avg Transaction: PHP 200-600
- Categories: Dining, Entertainment, Gaming
- Locations: Quezon City, Manila, Pasig

### 5. Occasional Users
- Age: 40-60 years
- Transaction Frequency: 5-15/month
- Avg Transaction: PHP 500-1,500
- Categories: Groceries, Healthcare, Transportation
- Locations: Cebu City, Davao City, Iloilo City, Bacolod

### 6. Business/Merchant Users
- Age: 30-50 years
- Transaction Frequency: 80-150/month
- Avg Transaction: PHP 5,000-15,000
- Categories: Business, Shopping, Wholesale
- Locations: Makati, BGC, Quezon City

### 7. Remittance Users
- Age: 25-45 years
- Transaction Frequency: 15-30/month
- Avg Transaction: PHP 3,000-8,000
- Categories: Remittance, Cash-out, Money Transfer
- Locations: Manila, Quezon City, Caloocan, Valenzuela

## Data Schema

### Customer Profile Fields

```json
{
  "customer_id": "string (16-char hash)",
  "age": "integer (18-60)",
  "location": "string (Philippine city)",
  "transaction_frequency": "integer (transactions per month)",
  "average_transaction_value": "float (PHP)",
  "merchant_categories": ["array of strings"],
  "total_spend": "float (PHP)",
  "account_age_days": "integer (30-1095)",
  "preferred_payment_methods": ["array of strings"],
  "last_transaction_date": "ISO 8601 datetime",
  "created_at": "ISO 8601 datetime",
  "updated_at": "ISO 8601 datetime"
}
```

### Transaction Fields

```json
{
  "transaction_id": "string (UUID)",
  "customer_id": "string (16-char hash)",
  "amount": "float (PHP)",
  "merchant_category": "string",
  "merchant_name": "string",
  "transaction_type": "string (payment|transfer|cashout)",
  "timestamp": "ISO 8601 datetime",
  "payment_method": "string",
  "location": "string (Philippine city)"
}
```

## Usage Examples

### Loading Customer Data (Python)

```python
import json
import pandas as pd

# Load JSON
with open('synthetic_data/synthetic_customers.json', 'r') as f:
    customers = json.load(f)

# Load CSV
df = pd.read_csv('synthetic_data/synthetic_customers.csv')

# Parse list columns from CSV
df['merchant_categories'] = df['merchant_categories'].str.split('|')
df['preferred_payment_methods'] = df['preferred_payment_methods'].str.split('|')
```

### Loading Transaction Data

```python
import json

with open('synthetic_data/synthetic_transactions_sample.json', 'r') as f:
    transactions = json.load(f)

# Group by customer
from collections import defaultdict
customer_transactions = defaultdict(list)
for txn in transactions:
    customer_transactions[txn['customer_id']].append(txn)
```

## Regenerating Data

To regenerate all datasets with a different seed or configuration:

```bash
python scripts/generate_synthetic_data.py
```

The generator uses:
- **Faker library** with Philippine locale (`en_PH`)
- **NumPy** for statistical distributions
- **Pandas** for data manipulation
- **Seed**: 42 (for reproducibility)

## Validation Checks

The main dataset has been validated for:

1. **Distribution Validation**
   - Age range: 18-60 years ✓
   - Transaction frequency: 1-315/month ✓
   - All required fields present ✓
   - No null values in required fields ✓

2. **Clustering Suitability**
   - PCA variance threshold: 80% explained by 3 components ✓
   - K-Means silhouette scores: 0.313-0.355 for k=3-10 ✓
   - Feature variance: Sufficient for clustering ✓
   - Feature correlations: Realistic patterns ✓

3. **Data Quality**
   - Realistic distributions across all features ✓
   - Natural clustering patterns present ✓
   - Outliers included (5% of dataset) ✓
   - Diverse geographic distribution ✓

## Testing Recommendations

### Unit Tests
- Use main dataset for standard test cases
- Use edge case dataset for boundary condition tests
- Use quality issues dataset for error handling tests

### Integration Tests
- Use main dataset for end-to-end workflow tests
- Use transaction sample for transaction analysis tests

### Performance Tests
- Use large dataset (50,000 customers) for scalability tests
- Target: Process 10,000 records in < 60 seconds

### Property-Based Tests
- Use Hypothesis with synthetic data generators
- Test invariants across all archetype combinations

## Notes

- All customer IDs are anonymized SHA-256 hashes
- Dates are in ISO 8601 format
- Currency values are in Philippine Pesos (PHP)
- Locations are Philippine cities
- The `archetype` field is included for validation only and should be removed before production use
