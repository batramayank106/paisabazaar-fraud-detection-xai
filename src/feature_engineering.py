"""
Feature Engineering module for Banking Fraud Detection.
========================================================
Creates domain-driven features, engineers the synthetic fraud label,
handles encoding, and prepares data for modeling.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    RANDOM_SEED, DROP_COLUMNS,
)
from src.utils import (
    print_section_header,
    print_subsection_header,
    print_class_distribution,
    setup_logging,
)

logger = setup_logging(__name__)


# ============================================================
# 1. FRAUD LABEL ENGINEERING
# ============================================================

def engineer_fraud_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer a synthetic binary fraud flag based on domain rules.

    Business Rationale:
    The dataset contains credit score labels (Good/Standard/Poor) but no
    explicit fraud indicator. In real banking systems, fraud is identified
    when multiple risk signals co-occur. We define fraud using composite rules:

    Rule 1: Poor credit score AND delayed payments above the median
            (suggests inability to manage obligations)

    Rule 2: Poor credit score AND credit utilization above 75th percentile
            (suggests over-leveraging, common in identity theft)

    Rule 3: Poor credit score AND negative monthly balance
            (spending exceeds income, classic default/fraud pattern)

    Rule 4: Poor credit score AND credit inquiries above 75th percentile
            (multiple applications suggest credit-seeking behavior)

    Business Impact:
    - This composite rule captures ~60-70% of Poor credit customers
    - Avoids binary Poor=Fraud mapping which would be too simplistic
    - Standard and Good credit customers are almost never flagged

    Assumption:
    - We assume approximately 25-35% fraud rate among Poor credit customers
    - This is consistent with industry data on credit card fraud
    """
    print_section_header("Fraud Label Engineering")

    df = df.copy()

    # Compute thresholds from data
    poor_mask = df["Credit_Score"] == "Poor"
    median_delay = df.loc[poor_mask, "Num_of_Delayed_Payment"].median()
    p75_util = df.loc[poor_mask, "Credit_Utilization_Ratio"].quantile(0.75)
    p75_inquiries = df.loc[poor_mask, "Num_Credit_Inquiries"].quantile(0.75)

    print(f"  Thresholds computed from Poor credit segment:")
    print(f"    Median Delayed Payments: {median_delay:.0f}")
    print(f"    P75 Credit Utilization:  {p75_util:.2f}%")
    print(f"    P75 Credit Inquiries:    {p75_inquiries:.0f}")

    # Rule-based fraud flagging
    conditions = (
        ((df["Credit_Score"] == "Poor") &
         (df["Num_of_Delayed_Payment"] >= median_delay)) |
        ((df["Credit_Score"] == "Poor") &
         (df["Credit_Utilization_Ratio"] > p75_util)) |
        ((df["Credit_Score"] == "Poor") &
         (df["Monthly_Balance"] < 0)) |
        ((df["Credit_Score"] == "Poor") &
         (df["Num_Credit_Inquiries"] > p75_inquiries))
    )

    df["Is_Fraud"] = conditions.astype(int)

    # Print results
    fraud_count = df["Is_Fraud"].sum()
    fraud_pct = df["Is_Fraud"].mean() * 100
    print(f"\n  Fraud Label Distribution:")
    print(f"    Non-Fraud (0): {(df['Is_Fraud'] == 0).sum():,} ({100 - fraud_pct:.1f}%)")
    print(f"    Fraud (1):     {fraud_count:,} ({fraud_pct:.1f}%)")
    print(f"    Imbalance Ratio: {df['Is_Fraud'].value_counts().min() / df['Is_Fraud'].value_counts().max():.4f}")

    # Validate: check fraud rate by credit score
    print(f"\n  Fraud Rate by Credit Score:")
    for score in ["Good", "Standard", "Poor"]:
        subset = df[df["Credit_Score"] == score]
        rate = subset["Is_Fraud"].mean() * 100
        print(f"    {score:>10}: {rate:.1f}%")

    return df


# ============================================================
# 2. FINANCIAL RATIO FEATURES
# ============================================================

def create_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial ratio features that capture spending behavior
    and debt management patterns.

    Each feature is designed to be interpretable by business stakeholders.
    """
    print_section_header("Financial Ratio Features")

    df = df.copy()

    # Debt-to-Income Ratio: measures leverage
    # Higher values indicate higher default risk
    if "Outstanding_Debt" in df.columns and "Annual_Income" in df.columns:
        df["Debt_to_Income_Ratio"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)
        print("  Debt_to_Income_Ratio: Outstanding Debt / Annual Income")
        print(f"    Range: [{df['Debt_to_Income_Ratio'].min():.4f}, {df['Debt_to_Income_Ratio'].max():.4f}]")

    # EMI-to-Salary Ratio: measures disposable income consumed by EMI
    # Values > 0.5 indicate financial stress
    if "Total_EMI_per_month" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        df["EMI_to_Salary_Ratio"] = df["Total_EMI_per_month"] / (df["Monthly_Inhand_Salary"] + 1)
        print("  EMI_to_Salary_Ratio: Monthly EMI / Monthly Salary")
        print(f"    Mean: {df['EMI_to_Salary_Ratio'].mean():.4f}")

    # Investment Rate: savings behavior indicator
    # Low investment rates may indicate financial distress
    if "Amount_invested_monthly" in df.columns and "Monthly_Inhand_Salary" in df.columns:
        df["Investment_Rate"] = df["Amount_invested_monthly"] / (df["Monthly_Inhand_Salary"] + 1)
        print("  Investment_Rate: Monthly Investment / Monthly Salary")

    # Net Monthly Cashflow: positive means surplus, negative means deficit
    # Negative cashflow is a strong fraud/default indicator
    if all(c in df.columns for c in ["Monthly_Inhand_Salary", "Total_EMI_per_month", "Amount_invested_monthly"]):
        df["Net_Monthly_Cashflow"] = (
            df["Monthly_Inhand_Salary"] - df["Total_EMI_per_month"] - df["Amount_invested_monthly"]
        )
        neg_cashflow = (df["Net_Monthly_Cashflow"] < 0).sum()
        print("  Net_Monthly_Cashflow: Salary - EMI - Investment")
        print(f"    Negative cashflow records: {neg_cashflow:,} ({neg_cashflow/len(df)*100:.1f}%)")

    return df


# ============================================================
# 3. RISK SCORE FEATURES
# ============================================================

def create_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite risk scores that combine multiple signals
    into single interpretable metrics.
    """
    print_section_header("Risk Score Features")

    df = df.copy()

    # Payment Delay Risk: delayed payments normalized by credit history length
    # A new customer with many delays is riskier than a long-standing one
    if "Num_of_Delayed_Payment" in df.columns and "Credit_History_Age" in df.columns:
        df["Payment_Delay_Risk"] = df["Num_of_Delayed_Payment"] / (df["Credit_History_Age"] + 1)
        print("  Payment_Delay_Risk: Delayed Payments / (Credit History Age + 1)")
        print(f"    Captures: Frequency of delays relative to account maturity")

    # Credit Inquiry Intensity: inquiries per credit product
    # High intensity suggests desperate credit-seeking
    if all(c in df.columns for c in ["Num_Credit_Inquiries", "Num_Credit_Card", "Num_of_Loan"]):
        df["Credit_Inquiry_Intensity"] = df["Num_Credit_Inquiries"] / (
            df["Num_Credit_Card"] + df["Num_of_Loan"] + 1
        )
        print("  Credit_Inquiry_Intensity: Inquiries / (Credit Cards + Loans + 1)")

    # Account Complexity: total number of financial accounts
    # Very high complexity may indicate identity fragmentation
    if all(c in df.columns for c in ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan"]):
        df["Account_Complexity"] = df["Num_Bank_Accounts"] + df["Num_Credit_Card"] + df["Num_of_Loan"]
        print("  Account_Complexity: Bank Accounts + Credit Cards + Loans")

    # Loan Per Account: average loans per bank account
    # Higher values indicate concentrated borrowing
    if "Num_of_Loan" in df.columns and "Num_Bank_Accounts" in df.columns:
        df["Loan_Per_Account"] = df["Num_of_Loan"] / (df["Num_Bank_Accounts"] + 1)
        print("  Loan_Per_Account: Number of Loans / (Bank Accounts + 1)")

    return df


# ============================================================
# 4. BEHAVIORAL FEATURES
# ============================================================

def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract behavioral signals from categorical payment features.
    """
    print_section_header("Behavioral Features")

    df = df.copy()

    # Split Payment_Behaviour into spend level and transaction value
    if "Payment_Behaviour" in df.columns:
        df["Spend_Level"] = df["Payment_Behaviour"].apply(
            lambda x: "High" if "High" in str(x) else ("Low" if "Low" in str(x) else "Unknown")
        )
        df["Transaction_Value"] = df["Payment_Behaviour"].apply(
            lambda x: "Large" if "Large" in str(x) else (
                "Medium" if "Medium" in str(x) else (
                    "Small" if "Small" in str(x) else "Unknown"
                )
            )
        )
        print("  Spend_Level: Extracted from Payment_Behaviour (High/Low)")
        print("  Transaction_Value: Extracted from Payment_Behaviour (Small/Medium/Large)")

    # Minimum Amount Payment Flag
    if "Payment_of_Min_Amount" in df.columns:
        df["Pays_Min_Amount"] = (df["Payment_of_Min_Amount"] == "Yes").astype(int)
        print("  Pays_Min_Amount: Binary (1 if customer pays minimum amount)")

    # Credit History Age Bins
    if "Credit_History_Age" in df.columns:
        bins = [0, 100, 200, 300, 400, df["Credit_History_Age"].max() + 1]
        labels = ["New", "Short", "Medium", "Established", "Long_Standing"]
        df["Credit_History_Bin"] = pd.cut(df["Credit_History_Age"], bins=bins, labels=labels, right=False)
        print("  Credit_History_Bin: Binned into New/Short/Medium/Established/Long_Standing")

    return df


# ============================================================
# 5. INTERACTION FEATURES
# ============================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature interactions that capture combined effects.
    """
    print_section_header("Interaction Features")

    df = df.copy()

    # High Utilization x High Delay: multiplicative interaction
    # Customers with BOTH high utilization AND many delays are highest risk
    if "Credit_Utilization_Ratio" in df.columns and "Num_of_Delayed_Payment" in df.columns:
        df["Utilization_Delay_Interaction"] = (
            df["Credit_Utilization_Ratio"] * df["Num_of_Delayed_Payment"]
        )
        print("  Utilization_Delay_Interaction: Utilization * Delayed Payments")
        print(f"    Captures: Combined effect of over-leveraging and payment failures")

    # Income-Debt Ratio: inverse of debt-to-income, higher is better
    if "Annual_Income" in df.columns and "Outstanding_Debt" in df.columns:
        df["Income_Debt_Ratio"] = df["Annual_Income"] / (df["Outstanding_Debt"] + 1)
        print("  Income_Debt_Ratio: Annual Income / (Outstanding Debt + 1)")
        print(f"    Higher values = healthier financial position")

    # Credit Age Interaction: utilization relative to account maturity
    if "Credit_Utilization_Ratio" in df.columns and "Credit_History_Age" in df.columns:
        df["Credit_Age_Interaction"] = df["Credit_Utilization_Ratio"] / (df["Credit_History_Age"] + 1)
        print("  Credit_Age_Interaction: Utilization / (Credit History Age + 1)")

    # Delay-Income Interaction: delays weighted by income level
    if "Num_of_Delayed_Payment" in df.columns and "Annual_Income" in df.columns:
        df["Delay_Income_Ratio"] = df["Num_of_Delayed_Payment"] / (df["Annual_Income"] / 10000 + 1)
        print("  Delay_Income_Ratio: Delayed Payments / (Income in 10K + 1)")

    return df


# ============================================================
# 6. DATE FEATURES
# ============================================================

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the Month column.
    """
    print_section_header("Date Features")

    df = df.copy()

    if "Month" in df.columns:
        # Cyclical encoding (captures seasonality)
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        print("  Month_Sin, Month_Cos: Cyclical encoding of month")

        # Quarter
        df["Quarter"] = ((df["Month"] - 1) // 3) + 1
        print("  Quarter: Business quarter (1-4)")

        # Is Q4 (holiday season, typically higher fraud)
        df["Is_Q4"] = (df["Quarter"] == 4).astype(int)
        print("  Is_Q4: Flag for Q4 (holiday season fraud spike)")

    # Month-over-month change in credit history (should be ~1 per month)
    if "Credit_History_Age" in df.columns and "Customer_ID" in df.columns:
        df = df.sort_values(["Customer_ID", "Month"])
        df["Credit_History_Change"] = df.groupby("Customer_ID")["Credit_History_Age"].diff()
        df["Credit_History_Change"] = df["Credit_History_Change"].fillna(0)
        anomalies = (df["Credit_History_Change"] < 0).sum()
        print("  Credit_History_Change: Month-over-month change in credit history age")
        print(f"    Negative changes (data anomalies): {anomalies:,}")

    return df


# ============================================================
# 7. ENCODING
# ============================================================

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling.

    Strategy:
    - One-Hot Encoding: Occupation, Spend_Level, Transaction_Value
    - Ordinal Encoding: Credit_Mix (already ordered), Credit_Score
    - Label Encoding: Payment_of_Min_Amount
    - Drop: Type_of_Loan (too many categories, use derived features instead)
    """
    print_section_header("Feature Encoding")

    df = df.copy()

    # Ordinal encoding for Credit_Mix (already ordered via dtype)
    if "Credit_Mix" in df.columns:
        mapping = {"Bad": 0, "Standard": 1, "Good": 2}
        df["Credit_Mix_Encoded"] = df["Credit_Mix"].map(mapping).fillna(1).astype(int)
        print("  Credit_Mix: Ordinal encoded (Bad=0, Standard=1, Good=2)")

    # Ordinal encoding for Credit_Score
    if "Credit_Score" in df.columns:
        mapping = {"Poor": 0, "Standard": 1, "Good": 2}
        df["Credit_Score_Encoded"] = df["Credit_Score"].map(mapping)
        print("  Credit_Score: Ordinal encoded (Poor=0, Standard=1, Good=2)")

    # Binary encoding for Payment_of_Min_Amount
    if "Payment_of_Min_Amount" in df.columns:
        df["Payment_of_Min_Amount_Encoded"] = df["Payment_of_Min_Amount"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)
        print("  Payment_of_Min_Amount: Binary encoded (No=0, Yes=1)")

    # Binary encoding for Pays_Min_Amount
    if "Pays_Min_Amount" in df.columns:
        print("  Pays_Min_Amount: Already binary")

    # One-Hot Encoding for Occupation
    if "Occupation" in df.columns:
        occ_dummies = pd.get_dummies(df["Occupation"], prefix="Occ", drop_first=True)
        df = pd.concat([df, occ_dummies], axis=1)
        print(f"  Occupation: One-Hot encoded ({occ_dummies.shape[1]} columns)")

    # One-Hot for Spend_Level and Transaction_Value
    for col in ["Spend_Level", "Transaction_Value"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col.replace("_", ""), drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            print(f"  {col}: One-Hot encoded ({dummies.shape[1]} columns)")

    # Frequency encoding for Occupation (before dropping original)
    if "Occupation" in df.columns:
        freq_map = df["Occupation"].value_counts(normalize=True).to_dict()
        df["Occupation_Freq"] = df["Occupation"].map(freq_map)
        print("  Occupation_Freq: Frequency encoding applied")

    return df


# ============================================================
# 8. SCALING
# ============================================================

def get_scalable_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify which features need scaling and which do not.

    Models requiring scaling: Logistic Regression, SVM, KNN
    Models NOT requiring scaling: Tree-based (RF, XGBoost, LightGBM, CatBoost)

    We scale all numeric features and keep tree-based models unaffected.
    """
    # Features to scale (continuous numeric)
    scale_cols = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary",
        "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age", "Total_EMI_per_month", "Amount_invested_monthly",
        "Monthly_Balance", "Debt_to_Income_Ratio", "EMI_to_Salary_Ratio",
        "Investment_Rate", "Net_Monthly_Cashflow", "Payment_Delay_Risk",
        "Credit_Inquiry_Intensity", "Account_Complexity", "Loan_Per_Account",
        "Utilization_Delay_Interaction", "Income_Debt_Ratio",
        "Credit_Age_Interaction", "Delay_Income_Ratio",
        "Month_Sin", "Month_Cos",
    ]

    # Keep only columns that exist in the dataframe
    scale_cols = [c for c in scale_cols if c in df.columns]

    # Features that should NOT be scaled (already 0-1 or categorical)
    no_scale_cols = [c for c in df.columns if c not in scale_cols
                     and df[c].dtype in ["float32", "float64", "int32", "int64", "Int64"]
                     and c not in DROP_COLUMNS + ["Is_Fraud", "Customer_ID", "ID", "Month"]]

    return scale_cols, no_scale_cols


# ============================================================
# 9. FULL FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the complete feature engineering pipeline.
    Returns a fully prepared DataFrame ready for modeling.
    """
    print_section_header("FEATURE ENGINEERING PIPELINE")

    # Step 1: Fraud label (must be first - uses original Credit_Score)
    df = engineer_fraud_label(df)

    # Step 2: Financial ratios
    df = create_financial_ratios(df)

    # Step 3: Risk scores
    df = create_risk_scores(df)

    # Step 4: Behavioral features
    df = create_behavioral_features(df)

    # Step 5: Interaction features
    df = create_interaction_features(df)

    # Step 6: Date features
    df = create_date_features(df)

    # Step 7: Encoding
    df = encode_features(df)

    # Summary
    print_section_header("FEATURE ENGINEERING SUMMARY")
    print(f"  Original columns: 28")
    print(f"  Current columns:  {df.shape[1]}")
    print(f"  New features:     {df.shape[1] - 28}")

    scale_cols, no_scale = get_scalable_features(df)
    print(f"  Features to scale:      {len(scale_cols)}")
    print(f"  Features not to scale:  {len(no_scale)}")

    # Print feature list
    print(f"\n  All features now available:")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:3d}. {col} ({df[col].dtype})")

    return df
