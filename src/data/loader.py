"""
Data loader for FreshRetailNet-50K from HuggingFace.
Handles loading, initial type casting, and schema validation.

Dataset: https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K
Paper: arXiv:2505.16319
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Expected columns in FreshRetailNet-50K
EXPECTED_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
    "dt",
    "sale_amount",
    "hours_sale",
    "stock_hour6_22_cnt",
    "hours_stock_status",
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]


def load_from_huggingface(
    dataset_name: str = "Dingdong-Inc/FreshRetailNet-50K",
    split: str = "train",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load FreshRetailNet-50K from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load ('train' or 'eval').
        cache_dir: Local cache directory for downloaded data.

    Returns:
        DataFrame with all columns from the dataset.
    """
    from datasets import load_dataset

    logger.info(f"Loading {dataset_name} split='{split}' from HuggingFace...")
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    df = dataset.to_pandas()
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate schema
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.warning(f"Missing expected columns: {missing}")

    return df


def load_from_parquet(path: str | Path) -> pd.DataFrame:
    """Load data from a local parquet file.

    Args:
        path: Path to parquet file.

    Returns:
        DataFrame with loaded data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to appropriate types for downstream processing.

    Args:
        df: Raw dataframe from FreshRetailNet-50K.

    Returns:
        DataFrame with properly typed columns.
    """
    df = df.copy()

    # Date column
    df["dt"] = pd.to_datetime(df["dt"])

    # Categorical columns
    cat_cols = [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Numeric columns
    numeric_cols = [
        "sale_amount",
        "stock_hour6_22_cnt",
        "discount",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean flags
    for col in ["holiday_flag", "activity_flag"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    logger.info("Type casting complete")
    return df


def create_series_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique series identifier for each store-product combination.

    Args:
        df: DataFrame with store_id and product_id columns.

    Returns:
        DataFrame with added 'series_id' column.
    """
    df = df.copy()
    df["series_id"] = (
        df["store_id"].astype(str) + "__" + df["product_id"].astype(str)
    )
    n_series = df["series_id"].nunique()
    logger.info(f"Created {n_series:,} unique series (store × product)")
    return df


def time_based_split(
    df: pd.DataFrame,
    test_days: int = 7,
    val_days: int = 7,
    date_col: str = "dt",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train/val/test.

    Per FRN-50K: 90 days per series. Last 7 = test, 7 before = val, rest = train.

    Args:
        df: Full dataset.
        test_days: Number of days for test set.
        val_days: Number of days for validation set.
        date_col: Name of the date column.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    test = df[df[date_col] >= test_start].copy()
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].copy()
    train = df[df[date_col] < val_start].copy()

    logger.info(
        f"Split: train={len(train):,} ({train[date_col].min()} to {train[date_col].max()}), "
        f"val={len(val):,} ({val[date_col].min()} to {val[date_col].max()}), "
        f"test={len(test):,} ({test[date_col].min()} to {test[date_col].max()})"
    )
    return train, val, test
