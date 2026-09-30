"""Dataset loading, inspection, and feature preparation helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MALL_CUSTOMERS_URL = (
    "https://raw.githubusercontent.com/devotuoma/"
    "Mall-Customers-Segmentation/main/Mall_Customers.csv"
)
CUSTOMER_FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def _make_offline_customers(random_state: int = 42) -> pd.DataFrame:
    """Build a clearly labeled, deterministic customer-style offline sample."""
    rng = np.random.default_rng(random_state)
    centers = [(25, 25, 75), (48, 25, 25), (33, 55, 50), (40, 85, 80), (52, 85, 20)]
    records: list[tuple[str, int, float, float, float]] = []
    customer_id = 1
    for age_center, income_center, score_center in centers:
        for _ in range(60):
            records.append(
                (
                    f"Customer {customer_id}",
                    str(rng.choice(["Male", "Female"])),
                    float(np.clip(rng.normal(age_center, 5), 18, 70)),
                    float(np.clip(rng.normal(income_center, 7), 15, 140)),
                    float(np.clip(rng.normal(score_center, 8), 1, 100)),
                )
            )
            customer_id += 1
    return pd.DataFrame(
        records,
        columns=["CustomerID", "Genre", *CUSTOMER_FEATURES],
    )


def load_customer_data(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load local Mall data, download the public CSV, or use an offline fallback."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "Mall_Customers.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path), "Public Mall Customers CSV (local copy)"

    try:
        with urlopen(MALL_CUSTOMERS_URL, timeout=20) as response:
            csv_path.write_bytes(response.read())
        return pd.read_csv(csv_path), "Public Mall Customers CSV (downloaded)"
    except (OSError, URLError, TimeoutError) as error:
        fallback_path = data_dir / "synthetic_customers_offline_fallback.csv"
        customers = _make_offline_customers()
        customers.to_csv(fallback_path, index=False)
        return customers, f"Synthetic offline fallback (download failed: {error})"


def inspect_dataset(data: pd.DataFrame, dataset_name: str) -> dict[str, object]:
    """Print and return the requested basic dataset checks."""
    print(f"\n{dataset_name} shape: {data.shape}")
    print(f"\n{dataset_name} first five rows:\n{data.head().to_string(index=False)}")
    missing = data.isna().sum()
    print(f"\nMissing values:\n{missing.to_string()}")
    duplicate_count = int(data.duplicated().sum())
    print(f"\nDuplicate records: {duplicate_count}")
    print(f"\nDescriptive statistics:\n{data.describe(include='all').to_string()}")
    return {"shape": data.shape, "missing": missing, "duplicates": duplicate_count}


def prepare_customer_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """Impute numeric customer features and standardize them for distance models."""
    absent = [feature for feature in CUSTOMER_FEATURES if feature not in data.columns]
    if absent:
        raise ValueError(f"Mall customer data is missing required features: {absent}")

    cleaned = data.drop_duplicates().copy()
    imputer = SimpleImputer(strategy="median")
    feature_values = imputer.fit_transform(cleaned[CUSTOMER_FEATURES])
    scaler = StandardScaler()
    standardized = scaler.fit_transform(feature_values)
    return cleaned, standardized, scaler