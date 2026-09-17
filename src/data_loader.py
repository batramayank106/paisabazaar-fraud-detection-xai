"""
Data loading and initial audit module.
=======================================
Handles CSV loading, initial profiling, memory optimization,
and basic integrity checks.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from src.config import DATA_PATH, RANDOM_SEED
from src.utils import (
    print_section_header,
    print_audit_report,
    print_missing_report,
    optimize_memory,
    setup_logging,
)

logger = setup_logging(__name__)


def load_data(path: str = None, optimize: bool = True) -> pd.DataFrame:
    """
    Load the dataset from CSV and perform initial audit.

    Parameters
    ----------
    path : str, optional
        Path to CSV file. Defaults to config.DATA_PATH.
    optimize : bool
        Whether to downcast numeric types to save memory.

    Returns
    -------
    pd.DataFrame
        Raw loaded dataframe.
    """
    load_path = path or DATA_PATH
    logger.info(f"Loading data from: {load_path}")

    df = pd.read_csv(load_path)
    logger.info(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    if optimize:
        df = optimize_memory(df)

    return df


def initial_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a comprehensive initial audit of the raw dataset.

    Returns a dictionary with key audit metrics for downstream use.
    """
    print_section_header("Initial Dataset Audit")

    audit = {
        "shape": df.shape,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
        "dtypes": df.dtypes.to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "duplicate_pct": float(df.duplicated().mean() * 100),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": float(df.isnull().mean().mean() * 100),
        "numeric_cols": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_cols": list(df.select_dtypes(include=["object", "category"]).columns),
        "n_customers": df["Customer_ID"].nunique() if "Customer_ID" in df.columns else None,
    }

    print_audit_report(df, "Raw Dataset")

    print("\n--- Summary Statistics (Numeric) ---")
    print(df.describe().round(2).to_string())

    print("\n--- Summary Statistics (Categorical) ---")
    cat_cols = audit["categorical_cols"]
    if cat_cols:
        print(df[cat_cols].describe().to_string())

    print("\n--- Missing Value Report ---")
    print_missing_report(df)

    return audit


def get_customer_panel_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since the dataset is panel data (multiple months per customer),
    compute customer-level aggregates to understand the structure.
    """
    if "Customer_ID" not in df.columns:
        logger.warning("Customer_ID column not found. Skipping panel analysis.")
        return pd.DataFrame()

    panel = df.groupby("Customer_ID").agg(
        n_months=("ID", "count"),
        first_month=("Month", "min"),
        last_month=("Month", "max"),
        unique_names=("Name", "nunique"),
        unique_ssns=("SSN", "nunique"),
    ).reset_index()

    print_subsection = "\n--- Customer Panel Structure ---\n"
    print(print_subsection)
    print(f"Total customers: {panel.shape[0]:,}")
    print(f"Avg months per customer: {panel['n_months'].mean():.1f}")
    print(f"Customers with all 8 months: {(panel['n_months'] == 8).sum():,}")
    print(f"Customers with name changes: {(panel['unique_names'] > 1).sum()}")
    print(f"Customers with SSN changes: {(panel['unique_ssns'] > 1).sum()}")

    return panel


def detect_special_characters(df: pd.DataFrame) -> Dict[str, list]:
    """
    Scan string columns for special characters, corrupted values,
    and sentinel patterns that indicate data quality issues.
    """
    print_section_header("Special Character Detection")
    issues = {}

    for col in df.select_dtypes(include=["object"]).columns:
        col_issues = []
        for val in df[col].dropna().unique():
            val_str = str(val)
            if any(c in val_str for c in "!@#$%^&*()_+{}|:<>?"):
                col_issues.append(val_str)
        if col_issues:
            issues[col] = col_issues
            print(f"  {col}: Found {len(col_issues)} problematic values")
            for v in col_issues[:5]:
                print(f"    -> '{v}'")

    if not issues:
        print("  No special character issues detected.")

    return issues
