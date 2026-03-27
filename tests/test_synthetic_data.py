"""Unit tests for synthetic data generation (Task 3.3)."""

import pytest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from src.services.synthetic_data_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Unit tests for SyntheticDataGenerator."""

    def setup_method(self):
        self.generator = SyntheticDataGenerator(seed=42)

    def test_archetype_count(self):
        """Verify all 7 archetypes are defined."""
        assert len(self.generator.archetypes) == 7

    def test_generate_customer_profile_fields(self):
        """Verify generated profiles have all required fields."""
        archetype = self.generator.archetypes["high_value_frequent"]
        profile = self.generator.generate_customer_profile(archetype)

        required_fields = [
            "customer_id", "age", "location", "transaction_frequency",
            "average_transaction_value", "merchant_categories", "total_spend",
            "account_age_days", "preferred_payment_methods", "last_transaction_date",
        ]
        for field in required_fields:
            assert field in profile, f"Missing field: {field}"

    def test_archetype_distribution_accuracy(self):
        """Test that dataset archetype distribution matches requested proportions."""
        distribution = {
            "high_value_frequent": 0.20,
            "budget_conscious": 0.30,
            "bill_payment": 0.10,
            "young_digital": 0.20,
            "occasional": 0.10,
            "business_merchant": 0.05,
            "remittance": 0.05,
        }
        df = self.generator.generate_dataset(
            total_customers=1000, archetype_distribution=distribution
        )

        # Check non-outlier distribution is roughly correct
        non_outlier = df[df["archetype"] != "outlier"]
        total_non_outlier = len(non_outlier)

        for archetype, expected_prop in distribution.items():
            actual_count = len(non_outlier[non_outlier["archetype"] == archetype])
            actual_prop = actual_count / total_non_outlier
            assert abs(actual_prop - expected_prop) < 0.05, (
                f"Archetype {archetype}: expected ~{expected_prop:.2f}, got {actual_prop:.2f}"
            )

    def test_data_validation_age_range(self):
        """Test that generated ages fall within archetype ranges."""
        for name, archetype in self.generator.archetypes.items():
            profile = self.generator.generate_customer_profile(archetype)
            assert archetype["age_range"][0] <= profile["age"] <= archetype["age_range"][1], (
                f"Age {profile['age']} out of range for {name}"
            )

    def test_data_validation_transaction_frequency(self):
        """Test that transaction frequency falls within archetype ranges."""
        for name, archetype in self.generator.archetypes.items():
            profile = self.generator.generate_customer_profile(archetype)
            assert archetype["transaction_freq"][0] <= profile["transaction_frequency"] <= archetype["transaction_freq"][1], (
                f"Transaction freq {profile['transaction_frequency']} out of range for {name}"
            )

    def test_data_validation_positive_values(self):
        """Test that numeric values are positive."""
        archetype = self.generator.archetypes["budget_conscious"]
        profile = self.generator.generate_customer_profile(archetype)

        assert profile["average_transaction_value"] > 0
        assert profile["total_spend"] > 0
        assert profile["account_age_days"] > 0

    def test_generate_dataset_size(self):
        """Test that generated dataset has approximately the right size."""
        df = self.generator.generate_dataset(total_customers=500)
        # Includes 5% outliers
        assert len(df) >= 475  # At least 95% of requested
        assert len(df) <= 550  # Allow some margin

    def test_generate_dataset_no_null_ids(self):
        """Test that all customer IDs are non-null and unique."""
        df = self.generator.generate_dataset(total_customers=200)
        assert df["customer_id"].notna().all()
        assert df["customer_id"].nunique() == len(df)

    def test_clustering_suitability_pca_variance(self):
        """Test that generated data has sufficient variance for PCA."""
        df = self.generator.generate_dataset(total_customers=500)

        features = df[["age", "transaction_frequency", "average_transaction_value",
                        "total_spend", "account_age_days"]].values

        pca = PCA()
        pca.fit(features)

        # First 3 components should explain at least 70% variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        assert cumulative_variance[2] >= 0.70, (
            f"First 3 PCA components explain only {cumulative_variance[2]:.2f} variance"
        )

    def test_clustering_suitability_silhouette(self):
        """Test that generated data has reasonable clustering structure."""
        df = self.generator.generate_dataset(total_customers=500)

        features = df[["age", "transaction_frequency", "average_transaction_value",
                        "total_spend", "account_age_days"]].values

        # Standardize
        from sklearn.preprocessing import StandardScaler
        features_scaled = StandardScaler().fit_transform(features)

        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)

        score = silhouette_score(features_pca, labels)
        assert score > 0.1, f"Silhouette score {score:.3f} too low for clustering"

    def test_generate_transaction_history(self):
        """Test transaction history generation."""
        archetype = self.generator.archetypes["high_value_frequent"]
        profile = self.generator.generate_customer_profile(archetype)
        transactions = self.generator.generate_transaction_history(profile, num_transactions=10)

        assert len(transactions) == 10
        for txn in transactions:
            assert "transaction_id" in txn
            assert "amount" in txn
            assert txn["amount"] > 0
            assert txn["customer_id"] == profile["customer_id"]

    def test_export_to_json(self, tmp_path):
        """Test JSON export."""
        df = self.generator.generate_dataset(total_customers=50)
        filepath = str(tmp_path / "test_export.json")
        self.generator.export_to_json(df, filepath)

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == len(df)

    def test_export_to_csv(self, tmp_path):
        """Test CSV export."""
        df = self.generator.generate_dataset(total_customers=50)
        filepath = str(tmp_path / "test_export.csv")
        self.generator.export_to_csv(df, filepath)

        loaded = pd.read_csv(filepath)
        assert len(loaded) == len(df)

    def test_quality_check_no_negative_values(self):
        """Test quality: no negative numeric values in generated data."""
        df = self.generator.generate_dataset(total_customers=500)
        numeric_cols = ["age", "transaction_frequency", "average_transaction_value",
                        "total_spend", "account_age_days"]
        for col in numeric_cols:
            assert (df[col] >= 0).all(), f"Negative values found in {col}"
