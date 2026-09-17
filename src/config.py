"""
Configuration file for the Banking Fraud Detection project.
===========================================================
Centralizes all constants, paths, column names, color palettes,
random seeds, and model hyperparameters.

Author: Mayank Batra
Date: December 2025
"""

import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# RANDOM SEED
# ============================================================
RANDOM_SEED = 42

# ============================================================
# DATA COLUMNS
# ============================================================
ID_COLUMN = "ID"
CUSTOMER_ID_COLUMN = "Customer_ID"
TARGET_COLUMN = "Is_Fraud"
ORIGINAL_TARGET = "Credit_Score"

# Numeric columns (original)
NUMERIC_COLUMNS = [
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
]

# Categorical columns (original)
CATEGORICAL_COLUMNS = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount",
    "Payment_Behaviour", "Type_of_Loan",
]

# Columns to drop before modeling (identifiers, names, SSN)
DROP_COLUMNS = ["ID", "Customer_ID", "Name", "SSN", "Month"]

# ============================================================
# FRAUD LABEL DEFINITION THRESHOLDS
# ============================================================
# These thresholds define the synthetic fraud label.
# A customer-month record is flagged as fraudulent if it meets
# a combination of credit risk indicators.
FRAUD_RULES = {
    "poor_credit": "Credit_Score == 'Poor'",
    "high_delay": "Num_of_Delayed_Payment >= MEDIAN_DELAY",
    "high_utilization": "Credit_Utilization_Ratio > P75_UTILIZATION",
    "negative_balance": "Monthly_Balance < 0",
    "high_inquiries": "Num_Credit_Inquiries > P75_INQUIRIES",
    "high_debt_ratio": "Debt_to_Income_Ratio > P75_DEBT_RATIO",
}

# ============================================================
# COLOR PALETTES
# ============================================================
COLORS = {
    "primary": "#1a1a2e",
    "secondary": "#16213e",
    "accent": "#0f3460",
    "highlight": "#e94560",
    "fraud": "#e94560",
    "non_fraud": "#0f3460",
    "good": "#2ecc71",
    "standard": "#f39c12",
    "poor": "#e74c3c",
}

SEABORN_PALETTE = "Set2"
PLOTLY_COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]

# ============================================================
# PLOT STYLES
# ============================================================
PLOT_CONFIG = {
    "figure.figsize": (12, 6),
    "figure.dpi": 100,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
}

PLOTLY_TEMPLATE = "plotly_white"

# ============================================================
# MODEL HYPERPARAMETER GRIDS
# ============================================================
MODEL_PARAMS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"],
        "max_iter": [1000],
        "random_state": [RANDOM_SEED],
    },
    "decision_tree": {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
        "random_state": [RANDOM_SEED],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "random_state": [RANDOM_SEED],
        "n_jobs": [-1],
    },
    "xgboost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
        "random_state": [RANDOM_SEED],
        "eval_metric": ["logloss"],
        "use_label_encoder": [False],
    },
    "lightgbm": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 50, 70],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_samples": [5, 10, 20, 50],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [0, 0.1, 1.0],
        "random_state": [RANDOM_SEED],
        "verbose": [-1],
    },
    "catboost": {
        "iterations": [200, 300, 500],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 7],
        "random_seed": [RANDOM_SEED],
        "verbose": [0],
    },
    "isolation_forest": {
        "n_estimators": [100, 200, 300],
        "contamination": [0.05, 0.1, 0.15, 0.2],
        "max_samples": ["auto", 0.7, 0.8],
        "random_state": [RANDOM_SEED],
    },
}

# ============================================================
# CROSS-VALIDATION SETTINGS
# ============================================================
CV_FOLDS = 5
CV_SCORING = "roc_auc"

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
TEST_SIZE = 0.2
STRATIFY = True

# ============================================================
# THRESHOLD FOR CLASSIFICATION
# ============================================================
DEFAULT_THRESHOLD = 0.5

# ============================================================
# LOGGING
# ============================================================
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
