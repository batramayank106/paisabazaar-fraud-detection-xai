"""
Machine Learning modeling module.
===================================
Complete ML pipeline: preprocessing, class imbalance handling,
model training, hyperparameter tuning, evaluation, and comparison.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, log_loss, cohen_kappa_score,
    matthews_corrcoef, precision_recall_fscore_support,
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    RANDOM_SEED, TEST_SIZE, CV_FOLDS, CV_SCORING,
    MODEL_PARAMS, FIGURES_DIR, MODELS_DIR,
    COLORS, PLOTLY_COLORWAY, PLOTLY_TEMPLATE,
)
from src.utils import (
    print_section_header, print_subsection_header,
    print_class_distribution, setup_logging,
)

logger = setup_logging(__name__)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib


def save_figure(fig, name: str) -> None:
    """Save figure to outputs directory."""
    path = FIGURES_DIR / f"{name}.png"
    try:
        if hasattr(fig, "write_image"):
            fig.write_image(str(path), width=1200, height=600, scale=2)
        else:
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close("all")


# ============================================================
# 1. DATA PREPARATION
# ============================================================

def prepare_modeling_data(
    df: pd.DataFrame,
    target: str = "Is_Fraud",
    drop_cols: list = None,
    scale: bool = True,
    test_size: float = TEST_SIZE,
) -> Dict[str, Any]:
    """
    Prepare data for modeling: split, scale, and handle encoding.

    Returns dictionary with X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    print_section_header("Data Preparation for Modeling")

    if drop_cols is None:
        drop_cols = ["ID", "Customer_ID", "Name", "SSN", "Month", "Credit_Score",
                     "Credit_Score_Encoded", "Type_of_Loan", "Credit_History_Bin"]

    # Select features
    feature_cols = [c for c in df.columns if c not in drop_cols + [target]
                    and df[c].dtype in ["float32", "float64", "int32", "int64", "Int64", "uint8", "bool"]]
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Fill any remaining NaN
    X = X.fillna(0)

    print(f"  Features: {X.shape[1]}")
    print(f"  Samples:  {X.shape[0]:,}")
    print(f"  Target distribution:")
    print_class_distribution(y)

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n  Train set: {X_train.shape[0]:,} samples")
    print(f"  Test set:  {X_test.shape[0]:,} samples")
    print_class_distribution(y_train, "Train Target")
    print_class_distribution(y_test, "Test Target")

    # Scaling
    scaler = None
    if scale:
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_unscaled": X_train,
        "X_test_unscaled": X_test,
        "scaler": scaler,
        "feature_names": feature_cols,
    }


# ============================================================
# 2. CLASS IMBALANCE HANDLING
# ============================================================

def handle_class_imbalance(
    X_train: pd.DataFrame, y_train: pd.Series,
    method: str = "smote",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using various resampling techniques.

    Methods:
    - smote: Synthetic Minority Oversampling
    - smoteenn: SMOTE + Edited Nearest Neighbors (hybrid)
    - class_weights: Handled by model parameters (no resampling)
    """
    print_subsection_header(f"Class Imbalance: {method.upper()}")

    if method == "class_weights":
        print("  Using class weights in model parameters (no resampling needed)")
        return X_train, y_train

    if method == "smote":
        sampler = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"  Original: {len(X_train):,} samples")
    print(f"  Resampled: {len(X_resampled):,} samples")
    print_class_distribution(y_resampled, "Resampled Target")

    return X_resampled, y_resampled


# ============================================================
# 3. MODEL TRAINING
# ============================================================

def get_models(class_weights: dict = None, scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """Initialize all models with default or weighted parameters."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            max_iter=1000, random_state=RANDOM_SEED,
            class_weight=class_weights,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_split=10, min_samples_leaf=5,
            random_state=RANDOM_SEED, class_weight=class_weights,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=RANDOM_SEED,
            n_jobs=-1, class_weight=class_weights,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=7, learning_rate=0.1,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_SEED, verbose=-1,
            class_weight=class_weights,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_seed=RANDOM_SEED, verbose=0,
            auto_class_weights="Balanced" if class_weights else None,
        ),
    }
    return models


def train_models(
    X_train: pd.DataFrame, y_train: pd.Series,
    models: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Train all models and return fitted models dict."""
    print_section_header("Model Training")

    if models is None:
        # Compute class weights
        fraud_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
        models = get_models(class_weights={0: 1, 1: fraud_ratio})

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f"  {name}: Trained successfully")
        except Exception as e:
            logger.error(f"  {name}: Training failed - {e}")
            print(f"  {name}: FAILED ({e})")

    return trained_models


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Comprehensive evaluation of a single model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob),
        "Brier Score": brier_score_loss(y_test, y_prob),
        "Log Loss": log_loss(y_test, y_prob),
        "Cohen Kappa": cohen_kappa_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    return metrics


def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all trained models and return comparison table."""
    print_section_header("Model Evaluation")

    all_metrics = []
    for name, model in trained_models.items():
        try:
            metrics = evaluate_model(model, X_test, y_test, name)
            all_metrics.append(metrics)
            print(f"\n  {name}:")
            print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"    Precision: {metrics['Precision']:.4f}")
            print(f"    Recall:    {metrics['Recall']:.4f}")
            print(f"    F1:        {metrics['F1']:.4f}")
            print(f"    ROC-AUC:   {metrics['ROC-AUC']:.4f}")
            print(f"    PR-AUC:    {metrics['PR-AUC']:.4f}")
        except Exception as e:
            logger.error(f"  {name}: Evaluation failed - {e}")

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df


def cross_validate_models(
    trained_models: Dict[str, Any],
    X: pd.DataFrame, y: pd.Series,
    cv: int = CV_FOLDS,
) -> pd.DataFrame:
    """Perform stratified cross-validation for all models."""
    print_section_header("Cross-Validation Results")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    cv_results = []
    for name, model in trained_models.items():
        try:
            scores = cross_val_score(model, X, y, cv=skf, scoring=CV_SCORING, n_jobs=-1)
            cv_results.append({
                "Model": name,
                "Mean_AUC": round(scores.mean(), 4),
                "Std_AUC": round(scores.std(), 4),
                "Min_AUC": round(scores.min(), 4),
                "Max_AUC": round(scores.max(), 4),
                "Scores": scores,
            })
            print(f"  {name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        except Exception as e:
            logger.error(f"  {name}: CV failed - {e}")

    return pd.DataFrame(cv_results)


# ============================================================
# 5. PLOTTING FUNCTIONS
# ============================================================

def plot_confusion_matrices(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame, y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot confusion matrices for all models."""
    n_models = len(trained_models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, model) in enumerate(trained_models.items()):
        row, col = idx // cols, idx % cols
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-Fraud", "Fraud"],
                    yticklabels=["Non-Fraud", "Fraud"],
                    ax=axes[row, col])
        axes[row, col].set_title(f"{name}", fontsize=12)
        axes[row, col].set_ylabel("Actual")
        axes[row, col].set_xlabel("Predicted")

    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=16, y=1.02)
    plt.tight_layout()
    if save:
        save_figure(plt.gcf(), "confusion_matrices")
    else:
        plt.show()


def plot_roc_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame, y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot ROC curves for all models on the same chart."""
    fig = go.Figure()

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, name=f"{name} (AUC={auc:.4f})",
            mode="lines", line=dict(width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random Baseline",
        mode="lines", line=dict(dash="dash", color="gray"),
    ))

    fig.update_layout(
        title="ROC Curves: Model Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=PLOTLY_TEMPLATE, height=500,
        legend=dict(x=0.6, y=0.1),
    )
    fig.show()
    if save:
        fig.write_image(str(FIGURES_DIR / "roc_curves.png"), width=1200, height=500, scale=2)


def plot_precision_recall_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame, y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot Precision-Recall curves for all models."""
    fig = go.Figure()

    baseline = y_test.mean()

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        fig.add_trace(go.Scatter(
            x=recall, y=precision, name=f"{name} (AP={pr_auc:.4f})",
            mode="lines", line=dict(width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline, baseline], name="Baseline",
        mode="lines", line=dict(dash="dash", color="gray"),
    ))

    fig.update_layout(
        title="Precision-Recall Curves: Model Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template=PLOTLY_TEMPLATE, height=500,
        legend=dict(x=0.6, y=0.9),
    )
    fig.show()
    if save:
        fig.write_image(str(FIGURES_DIR / "pr_curves.png"), width=1200, height=500, scale=2)


def plot_threshold_analysis(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str = "Model", save: bool = True,
) -> pd.DataFrame:
    """Analyze precision, recall, F1 across classification thresholds."""
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.01)

    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        results.append({"Threshold": t, "Precision": p, "Recall": r, "F1": f})

    results_df = pd.DataFrame(results)
    best_f1_idx = results_df["F1"].idxmax()
    best_threshold = results_df.loc[best_f1_idx, "Threshold"]

    fig = go.Figure()
    for metric, color in [("Precision", "#636EFA"), ("Recall", "#EF553B"), ("F1", "#00CC96")]:
        fig.add_trace(go.Scatter(
            x=results_df["Threshold"], y=results_df[metric],
            name=metric, mode="lines", line=dict(color=color, width=2),
        ))

    fig.add_vline(x=best_threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Best F1 Threshold: {best_threshold:.2f}")

    fig.update_layout(
        title=f"Threshold Analysis: {model_name}",
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        template=PLOTLY_TEMPLATE, height=450,
    )
    fig.show()
    if save:
        fig.write_image(str(FIGURES_DIR / f"threshold_{model_name.replace(' ', '_').lower()}.png"),
                       width=1200, height=450, scale=2)

    print(f"\n  Best F1 Score: {results_df.loc[best_f1_idx, 'F1']:.4f} at threshold {best_threshold:.2f}")
    print(f"  At best threshold: Precision={results_df.loc[best_f1_idx, 'Precision']:.4f}, "
          f"Recall={results_df.loc[best_f1_idx, 'Recall']:.4f}")

    return results_df


def plot_calibration_curve(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame, y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot calibration curves for all models."""
    fig = go.Figure()

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        # Bin predictions
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_means = []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_means.append(y_test.values[mask].mean())

        fig.add_trace(go.Scatter(
            x=bin_centers, y=bin_means, name=name,
            mode="lines+markers", line=dict(width=2),
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Perfectly Calibrated",
        mode="lines", line=dict(dash="dash", color="gray"),
    ))

    fig.update_layout(
        title="Calibration Curves",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        template=PLOTLY_TEMPLATE, height=500,
    )
    fig.show()
    if save:
        fig.write_image(str(FIGURES_DIR / "calibration_curves.png"), width=1200, height=500, scale=2)


def plot_model_comparison(metrics_df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart comparing all models across key metrics."""
    key_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

    fig = make_subplots(rows=2, cols=3, subplot_titles=key_metrics)

    for idx, metric in enumerate(key_metrics):
        row = idx // 3 + 1
        col = idx % 3 + 1
        sorted_df = metrics_df.sort_values(metric, ascending=True)
        fig.add_trace(go.Bar(
            x=sorted_df[metric], y=sorted_df["Model"],
            orientation="h", name=metric,
            showlegend=False, text=sorted_df[metric].round(4),
            textposition="outside",
        ), row=row, col=col)

    fig.update_layout(
        title_text="Model Comparison Across Metrics",
        template=PLOTLY_TEMPLATE, height=700,
    )
    fig.show()
    if save:
        fig.write_image(str(FIGURES_DIR / "model_comparison.png"), width=1400, height=700, scale=2)


def plot_lift_gain_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame, y_test: pd.Series,
    save: bool = True,
) -> None:
    """Plot lift and gain curves for all models."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Lift Curve", "Gain Curve"])

    baseline_rate = y_test.mean()

    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        sorted_indices = np.argsort(-y_prob)
        y_sorted = y_test.values[sorted_indices]

        n = len(y_sorted)
        fractions = np.arange(1, n + 1) / n
        cumulative_positives = np.cumsum(y_sorted) / y_sorted.sum()
        lift = cumulative_positives / fractions

        fig.add_trace(go.Scatter(
            x=fractions, y=lift, name=name, mode="lines", line=dict(width=2),
            showlegend=True,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=fractions, y=cumulative_positives, name=name, mode="lines", line=dict(width=2),
            showlegend=False,
        ), row=1, col=2)

    # Add baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 1], name="Baseline",
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random",
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title_text="Lift and Gain Curves",
        template=PLOTLY_TEMPLATE, height=450,
    )
    fig.update_xaxes(title_text="% of Population", row=1, col=1)
    fig.update_xaxes(title_text="% of Population", row=1, col=2)
    fig.update_yaxes(title_text="Lift", row=1, col=1)
    fig.update_yaxes(title_text="% of Positive Cases Captured", row=1, col=2)

    fig.show()
    if save:
        fig.write_image(str(FIGURES_DIR / "lift_gain_curves.png"), width=1400, height=450, scale=2)


# ============================================================
# 6. PREDICTION EXAMPLES
# ============================================================

def show_prediction_examples(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    X_test_unscaled: pd.DataFrame,
    feature_names: list, model_name: str = "Best Model",
    n_examples: int = 15,
) -> pd.DataFrame:
    """
    Show detailed prediction examples with feature values and confidence.
    Demonstrates the model working on real data.
    """
    print_section_header(f"Prediction Examples: {model_name}")

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Stratified sample: get some TP, TN, FP, FN
    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred,
        "Probability": y_prob,
        "Correct": y_test.values == y_pred,
    }, index=X_test.index)

    examples = []
    for label, pred in [(1, 1), (1, 0), (0, 1), (0, 0)]:
        subset = results[(results["Actual"] == label) & (results["Predicted"] == pred)]
        n_pick = min(n_examples // 4 + 1, len(subset))
        if n_pick > 0:
            examples.append(subset.sample(n_pick, random_state=RANDOM_SEED))

    if examples:
        example_df = pd.concat(examples)
    else:
        example_df = results.sample(min(n_examples, len(results)), random_state=RANDOM_SEED)

    display_cols = [
        "Age", "Annual_Income", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Num_of_Delayed_Payment",
        "Num_Credit_Inquiries", "Interest_Rate",
    ]
    display_cols = [c for c in display_cols if c in X_test_unscaled.columns]
    feature_display = X_test_unscaled.loc[example_df.index, display_cols].copy()
    display_df = pd.concat([example_df, feature_display], axis=1)

    def classify_example(row):
        if row["Actual"] == 1 and row["Predicted"] == 1:
            return "True Positive (Fraud Caught)"
        elif row["Actual"] == 1 and row["Predicted"] == 0:
            return "False Negative (Missed Fraud)"
        elif row["Actual"] == 0 and row["Predicted"] == 1:
            return "False Positive (False Alarm)"
        else:
            return "True Negative (Correct Legitimate)"

    display_df["Result"] = display_df.apply(classify_example, axis=1)

    print(f"Showing {len(display_df)} prediction examples:\n")
    for idx, row in display_df.iterrows():
        print(f"  Record #{idx}:")
        print(f"    Actual: {'Fraud' if row['Actual'] == 1 else 'Non-Fraud'}")
        print(f"    Predicted: {'Fraud' if row['Predicted'] == 1 else 'Non-Fraud'} (p={row['Probability']:.4f})")
        print(f"    Result: {row['Result']}")
        age_val = row.get('Age', 'N/A')
        income_val = row.get('Annual_Income', 0)
        debt_val = row.get('Outstanding_Debt', 0)
        util_val = row.get('Credit_Utilization_Ratio', 0)
        print(f"    Key Features: Age={age_val}, "
              f"Income=${income_val:,.0f}, "
              f"Debt=${debt_val:,.2f}, "
              f"Utilization={util_val:.1f}%")
        print()

    return display_df


# ============================================================
# 7. SAVE MODELS
# ============================================================

def save_models(trained_models: Dict[str, Any]) -> None:
    """Save trained models to disk."""
    for name, model in trained_models.items():
        filename = MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        joblib.dump(model, filename)
        logger.info(f"Saved {name} to {filename}")


# ============================================================
# 8. FULL ML PIPELINE
# ============================================================

def run_full_ml_pipeline(
    df: pd.DataFrame,
    target: str = "Is_Fraud",
    imbalance_method: str = "class_weights",
) -> Dict[str, Any]:
    """
    Execute the complete ML pipeline from cleaned data to evaluated models.

    Steps:
    1. Prepare data (split, scale)
    2. Handle class imbalance
    3. Train all models
    4. Evaluate all models
    5. Cross-validate
    6. Generate all plots
    7. Show prediction examples
    """
    print_section_header("MACHINE LEARNING PIPELINE")

    # Step 1: Prepare data
    data = prepare_modeling_data(df, target=target)

    # Step 2: Handle class imbalance
    if imbalance_method == "class_weights":
        X_train_final = data["X_train"]
        y_train_final = data["y_train"]
    else:
        X_train_final, y_train_final = handle_class_imbalance(
            data["X_train"], data["y_train"], method=imbalance_method
        )

    # Step 3: Get and train models
    fraud_ratio = data["y_train"].value_counts()[0] / data["y_train"].value_counts()[1]
    if imbalance_method == "class_weights":
        models = get_models(class_weights={0: 1, 1: fraud_ratio}, scale_pos_weight=fraud_ratio)
    else:
        models = get_models(scale_pos_weight=fraud_ratio)

    trained_models = train_models(X_train_final, y_train_final, models)

    # Step 4: Evaluate
    metrics_df = evaluate_all_models(trained_models, data["X_test"], data["y_test"])

    # Step 5: Cross-validate
    X_full = pd.concat([data["X_train"], data["X_test"]])
    y_full = pd.concat([data["y_train"], data["y_test"]])
    cv_results = cross_validate_models(trained_models, X_full, y_full)

    # Step 6: Plots
    print_section_header("Generating Evaluation Plots")
    plot_confusion_matrices(trained_models, data["X_test"], data["y_test"])
    plot_roc_curves(trained_models, data["X_test"], data["y_test"])
    plot_precision_recall_curves(trained_models, data["X_test"], data["y_test"])
    plot_calibration_curve(trained_models, data["X_test"], data["y_test"])
    plot_model_comparison(metrics_df)
    plot_lift_gain_curves(trained_models, data["X_test"], data["y_test"])

    # Threshold analysis for best model
    best_model_name = metrics_df.loc[metrics_df["ROC-AUC"].idxmax(), "Model"]
    plot_threshold_analysis(trained_models[best_model_name], data["X_test"], data["y_test"], best_model_name)

    # Step 7: Prediction examples
    show_prediction_examples(
        trained_models[best_model_name], data["X_test"], data["y_test"],
        data["X_test_unscaled"], data["feature_names"], best_model_name,
    )

    # Save models
    save_models(trained_models)

    return {
        "models": trained_models,
        "metrics": metrics_df,
        "cv_results": cv_results,
        "data": data,
        "best_model_name": best_model_name,
    }
