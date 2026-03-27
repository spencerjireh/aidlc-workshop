#!/usr/bin/env python3
"""
Script to generate synthetic test datasets for the customer segmentation system.

Generates:
- Main dataset (10,000 customers)
- Large dataset for performance testing (50,000 customers)
- Edge case dataset (500 customers)
- Data quality test dataset (1,000 customers with issues)
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.synthetic_data_generator import SyntheticDataGenerator
import pandas as pd
import numpy as np


def create_output_directory():
    """Create output directory for synthetic data."""
    output_dir = Path('synthetic_data')
    output_dir.mkdir(exist_ok=True)
    return output_dir


def generate_main_dataset(generator: SyntheticDataGenerator, output_dir: Path):
    """Generate main dataset (10,000 customers)."""
    print("\n" + "="*60)
    print("Generating Main Dataset (10,000 customers)")
    print("="*60)
    
    df = generator.generate_dataset(total_customers=10000)
    
    # Generate validation report
    report = generator.generate_validation_report(df)
    
    # Save report
    with open(output_dir / 'main_dataset_report.txt', 'w') as f:
        f.write(report)
    
    # Export to JSON and CSV
    generator.export_to_json(df, str(output_dir / 'synthetic_customers.json'))
    generator.export_to_csv(df, str(output_dir / 'synthetic_customers.csv'))
    
    print(f"\n✓ Main dataset generated successfully")
    return df


def generate_large_dataset(generator: SyntheticDataGenerator, output_dir: Path):
    """Generate large dataset for performance testing (50,000 customers)."""
    print("\n" + "="*60)
    print("Generating Large Dataset (50,000 customers)")
    print("="*60)
    
    df = generator.generate_dataset(total_customers=50000)
    
    # Export to JSON and CSV
    generator.export_to_json(df, str(output_dir / 'synthetic_customers_large.json'))
    generator.export_to_csv(df, str(output_dir / 'synthetic_customers_large.csv'))
    
    print(f"\n✓ Large dataset generated successfully")
    return df


def generate_edge_case_dataset(generator: SyntheticDataGenerator, output_dir: Path):
    """Generate edge case dataset (500 customers with boundary conditions)."""
    print("\n" + "="*60)
    print("Generating Edge Case Dataset (500 customers)")
    print("="*60)
    
    edge_cases = []
    
    # Very young users (minimum age)
    for _ in range(50):
        archetype = generator.archetypes['young_digital'].copy()
        archetype['age_range'] = (18, 19)
        customer = generator.generate_customer_profile(archetype)
        customer['archetype'] = 'edge_very_young'
        edge_cases.append(customer)
    
    # Very old users (maximum age)
    for _ in range(50):
        archetype = generator.archetypes['occasional'].copy()
        archetype['age_range'] = (59, 60)
        customer = generator.generate_customer_profile(archetype)
        customer['archetype'] = 'edge_very_old'
        edge_cases.append(customer)
    
    # Extremely high frequency users
    for _ in range(50):
        archetype = generator.archetypes['business_merchant'].copy()
        archetype['transaction_freq'] = (145, 150)
        archetype['avg_transaction'] = (10000, 15000)
        customer = generator.generate_customer_profile(archetype)
        customer['archetype'] = 'edge_high_frequency'
        edge_cases.append(customer)
    
    # Very low activity users
    for _ in range(50):
        archetype = generator.archetypes['occasional'].copy()
        archetype['transaction_freq'] = (1, 5)
        archetype['avg_transaction'] = (50, 200)
        customer = generator.generate_customer_profile(archetype)
        customer['archetype'] = 'edge_low_activity'
        edge_cases.append(customer)
    
    # Extreme transaction values (very high)
    for _ in range(50):
        archetype = generator.archetypes['business_merchant'].copy()
        archetype['avg_transaction'] = (20000, 50000)
        customer = generator.generate_customer_profile(archetype)
        customer['archetype'] = 'edge_high_value'
        edge_cases.append(customer)
    
    # Extreme transaction values (very low)
    for _ in range(50):
        archetype = generator.archetypes['young_digital'].copy()
        archetype['avg_transaction'] = (10, 100)
        customer = generator.generate_customer_profile(archetype)
        customer['archetype'] = 'edge_low_value'
        edge_cases.append(customer)
    
    # New accounts (very recent)
    for _ in range(50):
        archetype = generator.archetypes['young_digital'].copy()
        customer = generator.generate_customer_profile(archetype)
        customer['account_age_days'] = np.random.randint(1, 30)
        customer['archetype'] = 'edge_new_account'
        edge_cases.append(customer)
    
    # Old accounts (very established)
    for _ in range(50):
        archetype = generator.archetypes['high_value_frequent'].copy()
        customer = generator.generate_customer_profile(archetype)
        customer['account_age_days'] = np.random.randint(1000, 1095)
        customer['archetype'] = 'edge_old_account'
        edge_cases.append(customer)
    
    # Single category users
    for _ in range(50):
        archetype = generator.archetypes['bill_payment'].copy()
        customer = generator.generate_customer_profile(archetype)
        customer['merchant_categories'] = [customer['merchant_categories'][0]]
        customer['archetype'] = 'edge_single_category'
        edge_cases.append(customer)
    
    # Multi-category users (diverse)
    for _ in range(50):
        archetype = generator.archetypes['high_value_frequent'].copy()
        customer = generator.generate_customer_profile(archetype)
        customer['merchant_categories'] = list(np.random.choice(
            list(generator.merchant_categories.keys()),
            size=5,
            replace=False
        ))
        customer['archetype'] = 'edge_multi_category'
        edge_cases.append(customer)
    
    df = pd.DataFrame(edge_cases)
    
    # Export
    generator.export_to_json(df, str(output_dir / 'synthetic_customers_edge_cases.json'))
    generator.export_to_csv(df, str(output_dir / 'synthetic_customers_edge_cases.csv'))
    
    print(f"\n✓ Edge case dataset generated successfully")
    print(f"  Edge case distribution:")
    print(df['archetype'].value_counts())
    
    return df


def generate_quality_issues_dataset(generator: SyntheticDataGenerator, output_dir: Path):
    """Generate dataset with intentional data quality issues (1,000 customers)."""
    print("\n" + "="*60)
    print("Generating Data Quality Issues Dataset (1,000 customers)")
    print("="*60)
    
    # Generate base dataset
    df = generator.generate_dataset(total_customers=1000)
    
    # Introduce missing values in location (5%)
    missing_location_indices = df.sample(frac=0.05).index
    df.loc[missing_location_indices, 'location'] = None
    print(f"  Introduced {len(missing_location_indices)} missing location values")
    
    # Introduce invalid ages (2%)
    invalid_age_indices = df.sample(frac=0.02).index
    df.loc[invalid_age_indices, 'age'] = -1
    print(f"  Introduced {len(invalid_age_indices)} invalid age values")
    
    # Introduce negative transaction frequencies (2%)
    invalid_freq_indices = df.sample(frac=0.02).index
    df.loc[invalid_freq_indices, 'transaction_frequency'] = -10
    print(f"  Introduced {len(invalid_freq_indices)} negative transaction frequencies")
    
    # Introduce negative transaction values (2%)
    invalid_value_indices = df.sample(frac=0.02).index
    df.loc[invalid_value_indices, 'average_transaction_value'] = -500.0
    print(f"  Introduced {len(invalid_value_indices)} negative transaction values")
    
    # Introduce duplicate customer IDs (1%)
    duplicates = df.sample(frac=0.01)
    df = pd.concat([df, duplicates], ignore_index=True)
    print(f"  Introduced {len(duplicates)} duplicate customer records")
    
    # Introduce empty merchant categories (1%)
    empty_categories_indices = df.sample(frac=0.01).index
    for idx in empty_categories_indices:
        df.at[idx, 'merchant_categories'] = []
    print(f"  Introduced {len(empty_categories_indices)} empty merchant categories")
    
    # Export
    generator.export_to_json(df, str(output_dir / 'synthetic_customers_quality_issues.json'))
    generator.export_to_csv(df, str(output_dir / 'synthetic_customers_quality_issues.csv'))
    
    print(f"\n✓ Data quality issues dataset generated successfully")
    print(f"  Total records (including duplicates): {len(df)}")
    
    return df


def generate_transaction_sample(generator: SyntheticDataGenerator, main_df: pd.DataFrame, output_dir: Path):
    """Generate transaction history for sample customers."""
    print("\n" + "="*60)
    print("Generating Transaction History Sample")
    print("="*60)
    
    # Generate transactions for first 1000 customers
    all_transactions = []
    sample_customers = main_df.head(1000)
    
    for idx, customer in sample_customers.iterrows():
        transactions = generator.generate_transaction_history(customer.to_dict())
        all_transactions.extend(transactions)
    
    transactions_df = pd.DataFrame(all_transactions)
    transactions_df.to_json(
        str(output_dir / 'synthetic_transactions_sample.json'),
        orient='records',
        indent=2
    )
    
    print(f"✓ Generated {len(transactions_df)} transactions for {len(sample_customers)} customers")
    print(f"  Exported to synthetic_transactions_sample.json")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("Synthetic Data Generation Script")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate all datasets
    main_df = generate_main_dataset(generator, output_dir)
    generate_large_dataset(generator, output_dir)
    generate_edge_case_dataset(generator, output_dir)
    generate_quality_issues_dataset(generator, output_dir)
    generate_transaction_sample(generator, main_df, output_dir)
    
    print("\n" + "="*60)
    print("All datasets generated successfully!")
    print("="*60)
    print(f"\nGenerated files in {output_dir}:")
    for file in sorted(output_dir.glob('*')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
