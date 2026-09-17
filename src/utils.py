"""
Utility functions for the Banking Fraud Detection project.
==========================================================
Provides logging setup, random seed initialization,
memory optimization, and formatted printing helpers.

Author: Mayank Batra
Date: December 2025
"""

import logging
import random
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, LOG_FORMAT, LOG_DATE_FORMAT


def set_all_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import xgboost
        xgboost.set_config(**{"seed": seed})
    except Exception:
        pass
    warnings.filterwarnings("ignore")


def setup_logging(name: str = "fraud_detection") -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"Memory: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    return df


def print_section_header(title: str, width: int = 70) -> None:
    """Print a formatted section header for notebook readability."""
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width + "\n")


def print_subsection_header(title: str, width: int = 50) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---\n")


def print_audit_report(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a comprehensive data audit report."""
    print_section_header(f"Data Audit Report: {name}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Total Cells: {df.shape[0] * df.shape[1]:,}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"Duplicate Rows: {df.duplicated().sum():,} ({df.duplicated().mean() * 100:.2f}%)")
    print(f"Missing Values: {df.isnull().sum().sum():,} ({df.isnull().mean().mean() * 100:.2f}%)")

    print(f"\nColumn Types:")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count}")

    print(f"\nUnique Values:")
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique:,} ({n_unique / len(df) * 100:.1f}%)")

    print(f"\nFirst 3 Rows:")
    print(df.head(3).to_string())


def print_missing_report(df: pd.DataFrame) -> None:
    """Print a detailed missing value report."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct,
        "Data Type": df.dtypes,
    })
    report = report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    if len(report) == 0:
        print("No missing values found in any column.")
    else:
        print(report.to_string())


def print_class_distribution(y: pd.Series, name: str = "Target") -> None:
    """Print class distribution with percentages."""
    counts = y.value_counts()
    pcts = y.value_counts(normalize=True) * 100
    dist = pd.DataFrame({"Count": counts, "Percentage": pcts})
    print(f"\n{name} Distribution:")
    print(dist.to_string())
    ratio = counts.min() / counts.max()
    print(f"\nImbalance Ratio (min/max): {ratio:.4f}")
    if ratio < 0.1:
        print("WARNING: Severe class imbalance detected.")
    elif ratio < 0.4:
        print("NOTICE: Moderate class imbalance detected.")
    else:
        print("Classes are relatively balanced.")
