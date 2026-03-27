"""
Synthetic Data Generator for E-Wallet Customer Segmentation

Generates realistic synthetic customer data with natural clustering patterns
suitable for PCA and K-Means segmentation testing.
"""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from faker import Faker


class SyntheticDataGenerator:
    """
    Generates synthetic e-wallet customer data based on predefined archetypes.
    
    Supports 7 customer archetypes:
    - High-Value Frequent Users
    - Budget-Conscious Shoppers
    - Bill Payment Users
    - Young Digital Natives
    - Occasional Users
    - Business/Merchant Users
    - Remittance Users
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        Faker.seed(seed)
        self.fake = Faker(['en_PH', 'en_US'])  # Philippine locale
        
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
        
        # Define customer archetypes
        self.archetypes = {
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
    
    def generate_customer_profile(self, archetype: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a single customer profile based on archetype.
        
        Args:
            archetype: Dictionary containing archetype parameters
            
        Returns:
            Dictionary containing customer profile data
        """
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

    
    def generate_dataset(
        self,
        total_customers: int = 10000,
        archetype_distribution: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate complete synthetic dataset with configurable archetype distribution.
        
        Args:
            total_customers: Total number of customers to generate
            archetype_distribution: Dictionary mapping archetype names to proportions
                                  (must sum to <= 1.0)
        
        Returns:
            DataFrame containing generated customer profiles
        """
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
            archetype = self.archetypes[archetype_name]
            
            for _ in range(num_customers):
                customer = self.generate_customer_profile(archetype)
                customer['archetype'] = archetype_name  # For validation only
                customers.append(customer)
        
        # Add some noise/outliers (5% of dataset)
        num_outliers = int(total_customers * 0.05)
        for _ in range(num_outliers):
            # Random archetype
            archetype = self.archetypes[np.random.choice(list(self.archetypes.keys()))]
            customer = self.generate_customer_profile(archetype)
            
            # Add noise to make it an outlier
            customer['transaction_frequency'] = int(
                customer['transaction_frequency'] * np.random.uniform(0.3, 2.5)
            )
            customer['average_transaction_value'] = (
                customer['average_transaction_value'] * np.random.uniform(0.2, 3.0)
            )
            customer['archetype'] = 'outlier'
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_transaction_history(
        self,
        customer_profile: Dict[str, Any],
        num_transactions: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate transaction history for a customer.
        
        Args:
            customer_profile: Customer profile dictionary
            num_transactions: Number of transactions to generate (defaults to transaction_frequency)
        
        Returns:
            List of transaction dictionaries
        """
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
                'transaction_type': np.random.choice(
                    ['payment', 'transfer', 'cashout'],
                    p=[0.7, 0.2, 0.1]
                ),
                'timestamp': transaction_date.isoformat(),
                'payment_method': payment_method,
                'location': customer_profile['location']
            }
            transactions.append(transaction)
        
        return transactions
    
    def export_to_json(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export dataset to JSON format.
        
        Args:
            df: DataFrame to export
            filename: Output filename
        """
        # Remove archetype column (used for validation only)
        df_export = df.drop(columns=['archetype'], errors='ignore')
        df_export.to_json(filename, orient='records', indent=2)
        print(f"Exported {len(df_export)} customer profiles to {filename}")
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Export dataset to CSV format.
        
        Args:
            df: DataFrame to export
            filename: Output filename
        """
        # Remove archetype column
        df_export = df.drop(columns=['archetype'], errors='ignore').copy()
        
        # Flatten list columns for CSV
        df_export['merchant_categories'] = df_export['merchant_categories'].apply(
            lambda x: '|'.join(x)
        )
        df_export['preferred_payment_methods'] = df_export['preferred_payment_methods'].apply(
            lambda x: '|'.join(x)
        )
        
        df_export.to_csv(filename, index=False)
        print(f"Exported {len(df_export)} customer profiles to {filename}")
    
    def generate_validation_report(self, df: pd.DataFrame) -> str:
        """
        Generate report showing archetype distribution and clustering suitability.
        
        Args:
            df: DataFrame containing generated customer data
        
        Returns:
            String containing the validation report
        """
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        
        report_lines = []
        report_lines.append("\n=== Synthetic Data Generation Report ===\n")
        report_lines.append(f"Total Customers: {len(df)}")
        report_lines.append(f"\nArchetype Distribution:")
        report_lines.append(str(df['archetype'].value_counts()))
        
        report_lines.append(f"\n=== Feature Statistics ===")
        report_lines.append(f"\nAge Distribution:")
        report_lines.append(str(df['age'].describe()))
        
        report_lines.append(f"\nTransaction Frequency Distribution:")
        report_lines.append(str(df['transaction_frequency'].describe()))
        
        report_lines.append(f"\nAverage Transaction Value Distribution:")
        report_lines.append(str(df['average_transaction_value'].describe()))
        
        report_lines.append(f"\nLocation Distribution (Top 10):")
        report_lines.append(str(df['location'].value_counts().head(10)))
        
        report_lines.append(f"\n=== Clustering Suitability Check ===")
        
        # Check if data has sufficient variance for PCA
        numeric_features = [
            'age', 'transaction_frequency', 'average_transaction_value',
            'total_spend', 'account_age_days'
        ]
        feature_matrix = df[numeric_features].values
        
        report_lines.append(f"Feature variance:")
        for i, feature in enumerate(numeric_features):
            report_lines.append(f"  {feature}: {np.var(feature_matrix[:, i]):.2f}")
        
        # Check correlation between features
        report_lines.append(f"\nFeature correlations:")
        correlation_matrix = np.corrcoef(feature_matrix.T)
        for i, feature1 in enumerate(numeric_features):
            for j, feature2 in enumerate(numeric_features):
                if i < j:
                    report_lines.append(
                        f"  {feature1} <-> {feature2}: {correlation_matrix[i, j]:.3f}"
                    )
        
        # PCA variance check
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
        
        pca = PCA()
        pca.fit(X_scaled)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        
        report_lines.append(f"\n✓ PCA: {n_components_80} components explain 80% variance")
        report_lines.append(
            f"  Explained variance by component: {pca.explained_variance_ratio_[:5]}"
        )
        
        # K-Means clustering check
        silhouette_scores = []
        for k in range(3, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append((k, score))
        
        report_lines.append(f"\n✓ K-Means silhouette scores:")
        for k, score in silhouette_scores:
            report_lines.append(f"  k={k}: {score:.3f}")
        
        best_k = max(silhouette_scores, key=lambda x: x[1])
        report_lines.append(
            f"\n✓ Optimal k: {best_k[0]} (silhouette score: {best_k[1]:.3f})"
        )
        
        report = '\n'.join(report_lines)
        print(report)
        return report
