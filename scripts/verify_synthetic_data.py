#!/usr/bin/env python3
"""
Quick verification script to test SyntheticDataGenerator functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.synthetic_data_generator import SyntheticDataGenerator


def test_generator_initialization():
    """Test that generator initializes correctly."""
    print("Testing generator initialization...")
    gen = SyntheticDataGenerator(seed=42)
    assert len(gen.archetypes) == 7, "Should have 7 archetypes"
    assert len(gen.locations) == 19, "Should have 19 locations"
    print("✓ Generator initialized successfully")


def test_customer_profile_generation():
    """Test customer profile generation."""
    print("\nTesting customer profile generation...")
    gen = SyntheticDataGenerator(seed=42)
    
    archetype = gen.archetypes['high_value_frequent']
    customer = gen.generate_customer_profile(archetype)
    
    # Verify required fields
    required_fields = [
        'customer_id', 'age', 'location', 'transaction_frequency',
        'average_transaction_value', 'merchant_categories', 'total_spend',
        'account_age_days', 'preferred_payment_methods', 'last_transaction_date',
        'created_at', 'updated_at'
    ]
    
    for field in required_fields:
        assert field in customer, f"Missing field: {field}"
    
    # Verify data types and ranges
    assert isinstance(customer['age'], int), "Age should be integer"
    assert 30 <= customer['age'] <= 45, "Age should be in archetype range"
    assert isinstance(customer['transaction_frequency'], int), "Transaction frequency should be integer"
    assert 50 <= customer['transaction_frequency'] <= 100, "Transaction frequency should be in archetype range"
    assert isinstance(customer['merchant_categories'], list), "Merchant categories should be list"
    assert len(customer['merchant_categories']) > 0, "Should have at least one merchant category"
    
    print("✓ Customer profile generated successfully")
    print(f"  Sample customer ID: {customer['customer_id']}")
    print(f"  Age: {customer['age']}, Location: {customer['location']}")
    print(f"  Transaction frequency: {customer['transaction_frequency']}/month")
    print(f"  Avg transaction value: PHP {customer['average_transaction_value']:.2f}")


def test_dataset_generation():
    """Test dataset generation."""
    print("\nTesting dataset generation...")
    gen = SyntheticDataGenerator(seed=42)
    
    df = gen.generate_dataset(total_customers=100)
    
    assert len(df) > 100, "Should include outliers"
    assert 'archetype' in df.columns, "Should have archetype column"
    assert df['age'].notna().all(), "Age should not have null values"
    assert df['customer_id'].is_unique, "Customer IDs should be unique"
    
    print("✓ Dataset generated successfully")
    print(f"  Total customers: {len(df)}")
    print(f"  Archetypes: {df['archetype'].nunique()}")


def test_transaction_generation():
    """Test transaction history generation."""
    print("\nTesting transaction history generation...")
    gen = SyntheticDataGenerator(seed=42)
    
    archetype = gen.archetypes['young_digital']
    customer = gen.generate_customer_profile(archetype)
    
    transactions = gen.generate_transaction_history(customer, num_transactions=10)
    
    assert len(transactions) == 10, "Should generate 10 transactions"
    assert all('transaction_id' in txn for txn in transactions), "All transactions should have ID"
    assert all(txn['customer_id'] == customer['customer_id'] for txn in transactions), "All transactions should belong to customer"
    
    print("✓ Transaction history generated successfully")
    print(f"  Generated {len(transactions)} transactions")
    print(f"  Sample transaction: PHP {transactions[0]['amount']:.2f} at {transactions[0]['merchant_category']}")


def test_export_methods():
    """Test export methods."""
    print("\nTesting export methods...")
    gen = SyntheticDataGenerator(seed=42)
    
    df = gen.generate_dataset(total_customers=10)
    
    # Test JSON export
    test_json = Path('test_export.json')
    gen.export_to_json(df, str(test_json))
    assert test_json.exists(), "JSON file should be created"
    test_json.unlink()  # Clean up
    
    # Test CSV export
    test_csv = Path('test_export.csv')
    gen.export_to_csv(df, str(test_csv))
    assert test_csv.exists(), "CSV file should be created"
    test_csv.unlink()  # Clean up
    
    print("✓ Export methods work correctly")


def main():
    """Run all verification tests."""
    print("="*60)
    print("SyntheticDataGenerator Verification")
    print("="*60)
    
    try:
        test_generator_initialization()
        test_customer_profile_generation()
        test_dataset_generation()
        test_transaction_generation()
        test_export_methods()
        
        print("\n" + "="*60)
        print("All verification tests passed! ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
