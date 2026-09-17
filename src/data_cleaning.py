"""
Exhaustive data cleaning pipeline.
====================================
Handles missing values, duplicates, invalid data, type corrections,
outlier detection, and consistency checks. Every step is documented
with business rationale.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    FIGURES_DIR, RANDOM_SEED,
)
from src.utils import (
    print_section_header,
    print_subsection_header,
    print_missing_report,
    setup_logging,
)

logger = setup_logging(__name__)


# ============================================================
# 1. MISSING VALUE HANDLING
# ============================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a detailed missing value analysis report.

    Business Rationale:
    - Even if no NaN values exist, sentinel values like '_' or 'NM'
      may act as implicit missing indicators that the model would misinterpret.
    """
    print_section_header("Missing Value Analysis")

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values,
        "Dtype": df.dtypes.values,
    })
    report = report.sort_values("Missing %", ascending=False).reset_index(drop=True)

    total_missing = report["Missing Count"].sum()
    print(f"Total missing values across all columns: {total_missing:,}")
    if total_missing == 0:
        print("No NaN/null values found. Checking for sentinel values next...\n")

    print(report[report["Missing Count"] > 0].to_string(index=False))

    return report


def detect_sentinel_values(df: pd.DataFrame) -> Dict[str, list]:
    """
    Detect implicit missing values encoded as special characters,
    placeholder strings, or sentinel patterns.

    Common sentinels in financial data:
    - '_' or '!@9#%8' (corrupted entries)
    - 'NM' (Not Meaningful / Not Available)
    - Empty strings or whitespace-only strings
    - 'NA', 'N/A', 'null', 'None', 'undefined'
    """
    print_section_header("Sentinel Value Detection")

    sentinels = ["_", "!@9#%8", "NM", "NA", "N/A", "null", "None", "undefined", "", " "]
    found = {}

    for col in df.select_dtypes(include=["object"]).columns:
        col_sentinels = []
        for sentinel in sentinels:
            count = (df[col].astype(str).str.strip() == sentinel).sum()
            if count > 0:
                col_sentinels.append({"value": sentinel, "count": int(count)})
        if col_sentinels:
            found[col] = col_sentinels
            print(f"\n  Column: '{col}'")
            for s in col_sentinels:
                print(f"    Value '{s['value']}': {s['count']:,} occurrences ({s['count']/len(df)*100:.2f}%)")

    if not found:
        print("  No sentinel values detected.")

    return found


def clean_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sentinel values with proper NaN for consistent handling.

    Business Impact:
    - Leaving '_' in Payment_Behaviour would create a false category
    - 'NM' in Payment_of_Min_Amount represents unknown, not a valid category
    """
    print_section_header("Sentinel Value Cleaning")

    df = df.copy()
    replacements = {
        "Payment_Behaviour": {"!@9#%8": np.nan},
        "Payment_of_Min_Amount": {"NM": np.nan},
    }

    for col, mapping in replacements.items():
        if col in df.columns:
            before_nans = df[col].isnull().sum()
            df[col] = df[col].replace(mapping)
            after_nans = df[col].isnull().sum()
            added = after_nans - before_nans
            print(f"  {col}: Replaced {added:,} sentinel values with NaN")

    return df


# ============================================================
# 2. MISSING VALUE HANDLING
# ============================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using appropriate strategies per column.

    Strategy Rationale:
    - Payment_Behaviour: Mode imputation (categorical, low missingness)
    - Payment_of_Min_Amount: Mode imputation (categorical, low missingness)
    - Numeric columns: Median imputation (robust to outliers present in financial data)

    Why not drop rows?
    - Each row represents a customer-month observation
    - Dropping would reduce panel data continuity
    - Missingness is minimal (<0.5% for all columns)
    """
    print_section_header("Missing Value Treatment")

    df = df.copy()
    missing_before = df.isnull().sum().sum()

    # Categorical: Mode imputation
    cat_missing = ["Payment_Behaviour", "Payment_of_Min_Amount"]
    for col in cat_missing:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(mode_val)
            print(f"  {col}: Mode imputed {n_filled:,} values with '{mode_val}'")

    # Numeric: Median imputation (robust to outliers in financial data)
    num_cols_with_missing = df.columns[df.isnull().any()].tolist()
    for col in num_cols_with_missing:
        if df[col].dtype in ["float32", "float64", "int32", "int64"]:
            median_val = df[col].median()
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: Median imputed {n_filled:,} values with {median_val:.4f}")

    missing_after = df.isnull().sum().sum()
    print(f"\n  Total missing before: {missing_before:,}")
    print(f"  Total missing after:  {missing_after:,}")

    return df


# ============================================================
# 3. DUPLICATE HANDLING
# ============================================================

def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove exact duplicate rows.

    Business Rationale:
    - Exact duplicates (same ID) indicate data pipeline errors
    - Customer-level duplicates across months are NOT duplicates
      (panel data structure)
    """
    print_section_header("Duplicate Analysis")

    df = df.copy()
    n_before = len(df)

    # Exact duplicates (all columns identical)
    exact_dups = df.duplicated(keep="first").sum()
    print(f"  Exact duplicate rows: {exact_dups:,}")

    # Duplicate by ID (should be unique per record)
    if "ID" in df.columns:
        id_dup_count = df["ID"].duplicated().sum()
        print(f"  Duplicate IDs: {id_dup_count:,}")

    # Customer-month duplicates
    if "Customer_ID" in df.columns and "Month" in df.columns:
        cm_dups = df.duplicated(subset=["Customer_ID", "Month"], keep="first").sum()
        print(f"  Customer-Month duplicates: {cm_dups:,}")

    if exact_dups > 0:
        df = df.drop_duplicates(keep="first")
        print(f"\n  Removed {exact_dups:,} exact duplicate rows")
        print(f"  Records: {n_before:,} -> {len(df):,}")
    else:
        print("\n  No exact duplicates to remove.")

    return df


# ============================================================
# 4. DATA TYPE CORRECTIONS
# ============================================================

def correct_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct and optimize data types for each column.

    Changes:
    - Credit_Mix: object -> ordered categorical (Bad < Standard < Good)
    - Payment_of_Min_Amount: object -> categorical
    - Payment_Behaviour: object -> categorical
    - Occupation: object -> categorical
    - Float columns that should be integers: round and cast
    """
    print_section_header("Data Type Corrections")

    df = df.copy()

    # Categorical conversions
    cat_mappings = {
        "Credit_Mix": pd.CategoricalDtype(categories=["Bad", "Standard", "Good"], ordered=True),
        "Payment_of_Min_Amount": pd.CategoricalDtype(categories=["No", "Yes"], ordered=False),
    }

    for col, dtype in cat_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
            print(f"  {col}: Converted to ordered categorical")

    simple_cats = ["Occupation", "Payment_Behaviour"]
    for col in simple_cats:
        if col in df.columns:
            df[col] = df[col].astype("category")
            print(f"  {col}: Converted to category")

    # Float columns that represent discrete counts should be integers
    int_like_cols = [
        "Age", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
        "Credit_History_Age",
    ]
    for col in int_like_cols:
        if col in df.columns and df[col].dtype in ["float32", "float64"]:
            df[col] = df[col].round().astype("Int64")
            print(f"  {col}: Rounded to integer type")

    return df


# ============================================================
# 5. INVALID DATA DETECTION
# ============================================================

def detect_invalid_data(df: pd.DataFrame) -> Dict[str, list]:
    """
    Systematically detect invalid or impossible values.

    Checks:
    - Negative values where only positive makes sense
    - Ages outside reasonable banking range (14-100)
    - Account counts below 0
    - Monthly balance exceeding annual income
    - Interest rates outside plausible range (0-50%)
    """
    print_section_header("Invalid Data Detection")

    issues = {}

    # Age validation
    if "Age" in df.columns:
        young = (df["Age"] < 18).sum()
        old = (df["Age"] > 80).sum()
        if young > 0:
            issues["Age < 18"] = int(young)
            print(f"  Age < 18: {young:,} records (may be valid for joint accounts)")
        if old > 0:
            issues["Age > 80"] = int(old)
            print(f"  Age > 80: {old:,} records")

    # Non-negative checks
    non_negative_cols = [
        "Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan",
        "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
        "Annual_Income", "Monthly_Inhand_Salary",
        "Outstanding_Debt", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance",
    ]
    for col in non_negative_cols:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                issues[f"Negative {col}"] = int(neg)
                print(f"  Negative {col}: {neg:,} records")

    # Interest rate sanity
    if "Interest_Rate" in df.columns:
        high_ir = (df["Interest_Rate"] > 50).sum()
        if high_ir > 0:
            issues["Interest_Rate > 50%"] = int(high_ir)
            print(f"  Interest_Rate > 50%: {high_ir:,} records")

    # EMI exceeding income
    if "Total_EMI_per_month" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        over_emi = (df["Total_EMI_per_month"] > df["Monthly_Inhand_Salary"]).sum()
        if over_emi > 0:
            issues["EMI > Income"] = int(over_emi)
            print(f"  EMI > Monthly Income: {over_emi:,} records (potential fraud indicator)")

    # Credit utilization > 100%
    if "Credit_Utilization_Ratio" in df.columns:
        high_util = (df["Credit_Utilization_Ratio"] > 100).sum()
        if high_util > 0:
            issues["Utilization > 100%"] = int(high_util)
            print(f"  Credit Utilization > 100%: {high_util:,} records")

    if not issues:
        print("  No critical invalid data issues detected.")

    return issues


# ============================================================
# 6. OUTLIER DETECTION
# ============================================================

def detect_outliers_iqr(df: pd.DataFrame, columns: list = None,
                        factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using the IQR method.

    Business Rationale:
    - Financial data often contains legitimate high-value transactions
    - We flag outliers but do NOT blindly remove them
    - Outliers in fraud context may represent actual fraudulent behavior
    """
    if columns is None:
        columns = NUMERIC_COLUMNS

    outlier_report = {}
    for col in columns:
        if col in df.columns and df[col].dtype in ["float32", "float64", "Int64", "int64"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            outlier_report[col] = {
                "n_outliers": int(n_outliers),
                "pct": float(n_outliers / len(df) * 100),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }

    return pd.DataFrame(outlier_report).T


def detect_outliers_zscore(df: pd.DataFrame, columns: list = None,
                           threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers using Z-score method."""
    if columns is None:
        columns = NUMERIC_COLUMNS

    outlier_report = {}
    for col in columns:
        if col in df.columns and df[col].dtype in ["float32", "float64", "Int64", "int64"]:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                n_outliers = (z_scores > threshold).sum()
            else:
                n_outliers = 0
            outlier_report[col] = {
                "n_outliers": int(n_outliers),
                "pct": float(n_outliers / len(df) * 100),
            }

    return pd.DataFrame(outlier_report).T


def analyze_outlier_fraud_overlap(df: pd.DataFrame, outlier_cols: list = None) -> pd.DataFrame:
    """
    Analyze overlap between statistical outliers and fraud label.

    Business Impact:
    - If most fraud cases ARE outliers, the model can leverage this signal
    - If many fraud cases are NOT outliers, simple threshold rules will fail
    - This analysis justifies the need for ML over rule-based systems
    """
    if "Is_Fraud" not in df.columns:
        logger.warning("Is_Fraud column not found. Skipping overlap analysis.")
        return pd.DataFrame()

    if outlier_cols is None:
        outlier_cols = ["Num_of_Delayed_Payment", "Credit_Utilization_Ratio",
                        "Outstanding_Debt", "Num_Credit_Inquiries"]

    results = []
    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            is_outlier = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)

            fraud_outlier = (is_outlier & (df["Is_Fraud"] == 1)).sum()
            fraud_not_outlier = (~is_outlier & (df["Is_Fraud"] == 1)).sum()
            total_fraud = df["Is_Fraud"].sum()

            results.append({
                "Feature": col,
                "Total_Outliers": int(is_outlier.sum()),
                "Fraud_in_Outliers": int(fraud_outlier),
                "Fraud_Missed": int(fraud_not_outlier),
                "Fraud_Capture_Rate": f"{fraud_outlier / total_fraud * 100:.1f}%" if total_fraud > 0 else "N/A",
            })

    return pd.DataFrame(results)


# ============================================================
# 7. CONSISTENCY CHECKS
# ============================================================

def check_feature_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate logical relationships between features.

    Business Rationale:
    - Inconsistent data could indicate identity theft or data entry fraud
    - For example, EMI exceeding income is a red flag for default risk
    - Credit history age decreasing over time is data corruption
    """
    print_section_header("Feature Consistency Checks")

    inconsistencies = []

    # EMI to income ratio
    if "Total_EMI_per_month" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        over_emi = (df["Total_EMI_per_month"] > df["Monthly_Inhand_Salary"]).sum()
        inconsistencies.append({
            "Check": "EMI > Monthly Income",
            "Count": int(over_emi),
            "Pct": f"{over_emi / len(df) * 100:.2f}%",
            "Interpretation": "High default risk / potential fraud" if over_emi > 0 else "Clean",
        })

    # Annual income vs monthly salary consistency
    if "Annual_Income" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        expected_monthly = df["Annual_Income"] / 12
        ratio = df["Monthly_Inhand_Salary"] / expected_monthly
        inconsistent = ((ratio < 0.5) | (ratio > 2.0)).sum()
        inconsistencies.append({
            "Check": "Annual Income vs Monthly Salary ratio",
            "Count": int(inconsistent),
            "Pct": f"{inconsistent / len(df) * 100:.2f}%",
            "Interpretation": "May indicate tax deductions or data inconsistency",
        })

    # Account count validation
    if "Num_Bank_Accounts" in df.columns:
        zero_accounts = (df["Num_Bank_Accounts"] == 0).sum()
        high_accounts = (df["Num_Bank_Accounts"] > 10).sum()
        inconsistencies.append({
            "Check": "Zero bank accounts",
            "Count": int(zero_accounts),
            "Pct": f"{zero_accounts / len(df) * 100:.2f}%",
            "Interpretation": "Possible new customer or data error",
        })
        inconsistencies.append({
            "Check": "More than 10 bank accounts",
            "Count": int(high_accounts),
            "Pct": f"{high_accounts / len(df) * 100:.2f}%",
            "Interpretation": "Unusual - possible identity fraud" if high_accounts > 0 else "Clean",
        })

    report = pd.DataFrame(inconsistencies)
    print(report.to_string(index=False))
    return report


# ============================================================
# 8. CATEGORICAL CLEANING
# ============================================================

def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize categorical variables.

    Steps:
    - Strip whitespace from all string columns
    - Standardize capitalization
    - Merge equivalent categories
    - Handle rare categories
    """
    print_section_header("Categorical Cleaning")

    df = df.copy()

    # Strip whitespace from all object/category columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].dtype.name == "category":
            df[col] = df[col].cat.rename_categories(
                lambda x: x.strip() if isinstance(x, str) else x
            )
        else:
            df[col] = df[col].astype(str).str.strip()

    # Payment_Behaviour: standardize the cleaned labels
    if "Payment_Behaviour" in df.columns:
        df["Payment_Behaviour"] = df["Payment_Behaviour"].str.strip()
        valid_behaviors = [
            "Low_spent_Small_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Large_value_payments",
            "High_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
        ]
        mask = ~df["Payment_Behaviour"].isin(valid_behaviors)
        n_invalid = mask.sum()
        if n_invalid > 0:
            df.loc[mask, "Payment_Behaviour"] = df.loc[~mask, "Payment_Behaviour"].mode()[0]
            print(f"  Payment_Behaviour: Fixed {n_invalid:,} invalid entries")

    print(f"  All string columns: Whitespace stripped")
    print(f"  Credit_Mix categories: {df['Credit_Mix'].cat.categories.tolist() if 'Credit_Mix' in df.columns and hasattr(df['Credit_Mix'], 'cat') else 'N/A'}")

    return df


# ============================================================
# 9. FULL CLEANING PIPELINE
# ============================================================

def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete data cleaning pipeline in sequence.

    Returns cleaned DataFrame with before/after summary.
    """
    print_section_header("COMPLETE DATA CLEANING PIPELINE")
    shape_before = df.shape
    missing_before = df.isnull().sum().sum()

    # Step 1: Detect sentinels
    sentinels = detect_sentinel_values(df)

    # Step 2: Replace sentinels with NaN
    df = clean_sentinel_values(df)

    # Step 3: Handle remaining missing values
    df = handle_missing_values(df)

    # Step 4: Remove duplicates
    df = handle_duplicates(df)

    # Step 5: Correct data types
    df = correct_data_types(df)

    # Step 6: Detect invalid data
    invalid = detect_invalid_data(df)

    # Step 7: Clean categoricals
    df = clean_categoricals(df)

    # Step 8: Feature consistency checks
    consistency = check_feature_consistency(df)

    # Summary
    print_section_header("CLEANING SUMMARY")
    print(f"  Records:     {shape_before[0]:,} -> {df.shape[0]:,} ({shape_before[0] - df.shape[0]:,} removed)")
    print(f"  Columns:     {shape_before[1]} -> {df.shape[1]}")
    print(f"  Missing:     {missing_before:,} -> {df.isnull().sum().sum():,}")
    print(f"  Sentinels:   {sum(s['count'] for v in sentinels.values() for s in v):,} replaced")
    print(f"  Invalid:     {sum(invalid.values()):,} issues detected (flagged, not removed)")

    return df
