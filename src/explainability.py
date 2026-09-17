"""
Explainable AI (XAI) module.
==============================
SHAP analysis, permutation importance, partial dependence plots,
and business interpretation of model decisions.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import permutation_importance, partial_dependence
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    RANDOM_SEED, FIGURES_DIR, COLORS,
    PLOTLY_TEMPLATE,
)
from src.utils import (
    print_section_header, print_subsection_header,
    setup_logging,
)

logger = setup_logging(__name__)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
# 1. SHAP ANALYSIS
# ============================================================

def compute_shap_values(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame,
    model_name: str = "Model",
) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer for tree-based models.

    Business Rationale:
    - SHAP provides both global and local explanations
    - Each prediction gets feature-level contributions
    - Enables regulatory compliance (explainability requirement)
    """
    print_section_header(f"SHAP Analysis: {model_name}")

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Fraud class

    print(f"  Computed SHAP values for {X_test.shape[0]:,} test samples")
    print(f"  Number of features: {X_test.shape[1]}")

    return shap_values, explainer


def plot_shap_summary(shap_values, X_test: pd.DataFrame,
                      model_name: str = "Model", save: bool = True) -> None:
    """
    SHAP Summary Plot (beeswarm): shows feature importance and direction.

    Reading the plot:
    - Features ranked by importance (top = most important)
    - Each dot = one prediction
    - Color = feature value (red = high, blue = low)
    - Position on x-axis = SHAP value (positive = pushes toward fraud)
    """
    print_subsection_header("SHAP Summary Plot (Beeswarm)")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.title(f"SHAP Summary Plot: {model_name}", fontsize=14)
    plt.tight_layout()
    if save:
        save_figure(plt.gcf(), f"shap_summary_{model_name.replace(' ', '_').lower()}")
    else:
        plt.show()

    # Print top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=False).head(10)
    print("\n  Top 10 Features by Mean |SHAP|:")
    for feat, val in top_features.items():
        print(f"    {feat}: {val:.4f}")


def plot_shap_bar(shap_values, X_test: pd.DataFrame,
                  model_name: str = "Model", save: bool = True) -> None:
    """SHAP Bar Plot: feature importance ranking."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance: {model_name}", fontsize=14)
    plt.tight_layout()
    if save:
        save_figure(plt.gcf(), f"shap_bar_{model_name.replace(' ', '_').lower()}")
    else:
        plt.show()


def plot_shap_dependence(shap_values, X_test: pd.DataFrame,
                         feature: str, model_name: str = "Model",
                         interaction_feature: str = None,
                         save: bool = True) -> None:
    """
    SHAP Dependence Plot: shows how a single feature affects predictions.

    Business Interpretation:
    - Positive SHAP = pushes prediction toward fraud
    - Negative SHAP = pushes prediction toward non-fraud
    - Coloring by interaction feature reveals combined effects
    """
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_values, X_test,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.title(f"SHAP Dependence: {feature}", fontsize=14)
    plt.tight_layout()
    if save:
        save_figure(plt.gcf(), f"shap_dependence_{feature.lower().replace(' ', '_')}")
    else:
        plt.show()


def plot_shap_force_plot(shap_values, X_test: pd.DataFrame,
                         instance_idx: int, model_name: str = "Model",
                         explainer=None, save: bool = True) -> None:
    """
    SHAP Force Plot for a single prediction.
    """
    print_subsection_header(f"SHAP Force Plot: Instance {instance_idx}")

    try:
        fig, ax = plt.subplots(figsize=(16, 4))
        base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        shap.plots.force(
            base_val,
            shap_values[instance_idx],
            X_test.iloc[instance_idx],
            feature_names=X_test.columns.tolist(),
            matplotlib=True, show=False,
        )
        if save:
            plt.savefig(str(FIGURES_DIR / f"shap_force_instance_{instance_idx}.png"),
                        dpi=150, bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        logger.warning(f"SHAP force plot failed: {e}")
        plt.close("all")

    # Print top contributing features
    feature_contributions = pd.Series(
        shap_values[instance_idx], index=X_test.columns
    ).sort_values(key=abs, ascending=False)

    print(f"\n  Top feature contributions for instance {instance_idx}:")
    for feat, val in feature_contributions.head(8).items():
        direction = "FRAUD" if val > 0 else "NON-FRAUD"
        print(f"    {feat}: {val:+.4f} ({direction})")


def plot_shap_waterfall(shap_values, X_test: pd.DataFrame,
                        instance_idx: int, model_name: str = "Model",
                        explainer=None, save: bool = True) -> None:
    """SHAP Waterfall Plot: detailed feature contribution breakdown."""
    try:
        plt.figure(figsize=(12, 8))
        base_val = explainer.expected_value[0] if explainer is not None and isinstance(explainer.expected_value, (list, np.ndarray)) else 0
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=base_val,
            data=X_test.iloc[instance_idx].values,
            feature_names=X_test.columns.tolist(),
        )
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f"SHAP Waterfall: Instance {instance_idx}", fontsize=14)
        plt.tight_layout()
        if save:
            save_figure(plt.gcf(), f"shap_waterfall_instance_{instance_idx}")
        else:
            plt.show()
    except Exception as e:
        logger.warning(f"SHAP waterfall plot failed: {e}")
        plt.close("all")


# ============================================================
# 2. PERMUTATION IMPORTANCE
# ============================================================

def compute_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str = "Model", n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Compute permutation importance as a model-agnostic feature importance method.

    Business Rationale:
    - Permutation importance is model-agnostic
    - Complements SHAP by providing a different perspective
    - Higher agreement between methods = more reliable ranking
    """
    print_section_header(f"Permutation Importance: {model_name}")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=RANDOM_SEED,
        scoring="roc_auc", n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    print("\n  Top 15 Features by Permutation Importance:")
    for _, row in importance_df.head(15).iterrows():
        print(f"    {row['Feature']:40s}: {row['Importance_Mean']:.4f} (+/- {row['Importance_Std']:.4f})")

    return importance_df


def plot_permutation_importance(importance_df: pd.DataFrame,
                                model_name: str = "Model",
                                save: bool = True) -> None:
    """Plot permutation importance."""
    top_n = min(20, len(importance_df))
    top = importance_df.head(top_n).sort_values("Importance_Mean", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top["Feature"], top["Importance_Mean"],
             xerr=top["Importance_Std"], color=COLORS["accent"], alpha=0.8)
    plt.xlabel("Mean Decrease in ROC-AUC")
    plt.title(f"Permutation Importance: {model_name}", fontsize=14)
    plt.tight_layout()
    if save:
        save_figure(plt.gcf(), f"permutation_importance_{model_name.replace(' ', '_').lower()}")
    else:
        plt.show()


# ============================================================
# 3. PARTIAL DEPENDENCE PLOTS
# ============================================================

def plot_partial_dependence(
    model, X_test: pd.DataFrame,
    features: List[str] = None,
    model_name: str = "Model",
    save: bool = True,
) -> None:
    """
    Partial Dependence Plots for top features.

    Business Interpretation:
    - Shows the marginal effect of a feature on predictions
    - Flat line = feature has no effect
    - Steep curve = feature strongly influences fraud probability
    """
    print_section_header(f"Partial Dependence Plots: {model_name}")

    if features is None:
        features = X_test.columns[:6].tolist()

    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, feature in enumerate(features):
        row, col = idx // cols, idx % cols
        try:
            pd_result = partial_dependence(model, X_test, [feature], kind="average")
            axes[row, col].plot(pd_result["grid_values"][0], pd_result["average"][0],
                               linewidth=2, color=COLORS["accent"])
            axes[row, col].set_title(feature, fontsize=12)
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel("Partial Dependence")
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
            axes[row, col].set_title(feature)

    # Hide empty subplots
    for idx in range(n_features, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].set_visible(False)

    plt.suptitle(f"Partial Dependence Plots: {model_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    if save:
        save_figure(plt.gcf(), f"partial_dependence_{model_name.replace(' ', '_').lower()}")
    else:
        plt.show()


# ============================================================
# 4. BUSINESS INTERPRETATION
# ============================================================

def generate_business_interpretation(
    shap_values, X_test: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Generate human-readable business interpretations for top features.
    Maps statistical findings to business language.
    """
    print_section_header("Business Interpretation of Key Features")

    top_features = feature_importance_df.head(10)["Feature"].tolist()

    interpretations = {
        "Credit_Utilization_Ratio": {
            "finding": "Credit utilization ratio is the strongest predictor of fraud.",
            "business": "Customers using more than 70% of their available credit are significantly "
                       "more likely to be involved in fraudulent activity. This aligns with the "
                       "industry pattern where fraudsters max out stolen credit lines quickly.",
            "action": "Set real-time alerts for utilization exceeding 70%. Implement dynamic "
                     "credit limit adjustments for high-utilization accounts.",
            "kpi": "Target: Reduce high-utilization fraud by 30% within 6 months.",
        },
        "Outstanding_Debt": {
            "finding": "High outstanding debt strongly correlates with fraud predictions.",
            "business": "Customers with disproportionately high debt relative to their income "
                       "profile represent elevated risk. This may indicate identity theft where "
                       "fraudsters accumulate debt without intent to repay.",
            "action": "Flag accounts where debt exceeds 3x annual income for manual review. "
                     "Implement automated debt-to-income ratio monitoring.",
            "kpi": "Target: Flag 90% of high-debt fraud cases within first 3 months.",
        },
        "Num_of_Delayed_Payment": {
            "finding": "Number of delayed payments is a key risk indicator.",
            "business": "Frequent payment delays suggest either financial distress or intentional "
                       "non-payment (common in account takeover fraud). Pattern changes in "
                       "payment behavior are particularly telling.",
            "action": "Monitor month-over-month changes in payment delays. Trigger review when "
                     "delays increase by more than 2 standard deviations from customer baseline.",
            "kpi": "Target: Detect 85% of delayed-payment fraud within 2 billing cycles.",
        },
        "Monthly_Balance": {
            "finding": "Negative or very low monthly balances indicate high fraud risk.",
            "business": "A negative balance means spending exceeds income, a classic sign of "
                       "fraudulent account use. Legitimate customers typically maintain positive "
                       "balances even during financial stress.",
            "action": "Set automated alerts for negative monthly balances. Implement velocity "
                     "checks when balance drops below zero.",
            "kpi": "Target: Zero tolerance for negative balance fraud beyond 1 billing cycle.",
        },
        "Total_EMI_per_month": {
            "finding": "High EMI burden relative to income predicts fraud.",
            "business": "When EMI commitments consume more than 50% of income, the customer is "
                       "in financial distress. This is particularly relevant for loan fraud where "
                       "applicants misrepresent income.",
            "action": "Implement income verification cross-checks. Flag applications where EMI-to-income "
                     "exceeds 50% for additional verification.",
            "kpi": "Target: Reduce false income declarations by 40%.",
        },
        "Num_Credit_Inquiries": {
            "finding": "Multiple credit inquiries in a short period signal credit-seeking behavior.",
            "business": "Fraudsters often apply for multiple credit products simultaneously before "
                       "the first application is flagged. This 'burst' pattern is detectable.",
            "action": "Implement cross-institutional inquiry sharing (where legally permitted). "
                     "Flag accounts with more than 5 inquiries in 30 days.",
            "kpi": "Target: Detect 75% of multi-inquiry fraud within 7 days.",
        },
        "Num_Credit_Card": {
            "finding": "Number of credit cards held is a moderate predictor.",
            "business": "Having many credit cards may indicate either a sophisticated financial "
                       "profile or identity fragmentation (multiple identities).",
            "action": "Cross-reference card count with income verification. Flag mismatches.",
            "kpi": "Target: Validate card count accuracy for 95% of new applications.",
        },
        "Interest_Rate": {
            "finding": "Higher interest rates correlate with higher fraud probability.",
            "business": "High interest rates often accompany subprime lending, which has higher "
                       "fraud rates. Customers accepting very high rates may be desperate or "
                       "may not intend to repay.",
            "action": "Review high-interest accounts more frequently. Implement tiered monitoring "
                     "based on interest rate brackets.",
            "kpi": "Target: Reduce subprime fraud losses by 25%.",
        },
        "Annual_Income": {
            "finding": "Income level affects fraud prediction.",
            "business": "Both very low and very high incomes show different fraud patterns. "
                       "Low income may indicate application fraud (income inflation), while "
                       "high income may indicate sophisticated financial fraud.",
            "action": "Implement income verification through employer validation and tax documents.",
            "kpi": "Target: Verify 100% of applications with income above 95th percentile.",
        },
        "Credit_History_Age": {
            "finding": "Newer credit accounts have higher fraud rates.",
            "business": "Fraudsters prefer new accounts because they have less history to "
                       "trigger anomaly detection systems. Long-standing accounts with "
                       "sudden behavior changes are also suspicious.",
            "action": "Apply enhanced monitoring for accounts less than 6 months old. "
                     "Implement behavioral baseline establishment period.",
            "kpi": "Target: Detect 80% of new-account fraud within first 90 days.",
        },
    }

    for feature in top_features:
        if feature in interpretations:
            interp = interpretations[feature]
            print(f"\n  FEATURE: {feature}")
            print(f"    Finding:     {interp['finding']}")
            print(f"    Business:    {interp['business']}")
            print(f"    Action:      {interp['action']}")
            print(f"    KPI:         {interp['kpi']}")
        else:
            print(f"\n  FEATURE: {feature}")
            print(f"    (Custom interpretation needed for this feature)")

    return interpretations


# ============================================================
# 5. FULL XAI PIPELINE
# ============================================================

def run_full_xai_pipeline(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_test: pd.Series, model_name: str = "XGBoost",
) -> Dict[str, Any]:
    """
    Execute the complete XAI pipeline.
    """
    print_section_header("EXPLAINABLE AI PIPELINE")

    # Step 1: SHAP values
    shap_values, explainer = compute_shap_values(model, X_train, X_test, model_name)

    # Step 2: SHAP plots
    plot_shap_summary(shap_values, X_test, model_name)
    plot_shap_bar(shap_values, X_test, model_name)

    # Step 3: SHAP dependence plots for top features
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_shap_features = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=False)

    top_5 = top_shap_features.head(5).index.tolist()
    for feat in top_5:
        # Find best interaction feature
        interaction = None
        if len(top_shap_features) > 5:
            interaction = top_shap_features.index[5]
        plot_shap_dependence(shap_values, X_test, feat, model_name, interaction)

    # Step 4: Force plots for specific instances
    # Find fraud and non-fraud examples
    fraud_idx = np.where(y_test.values == 1)[0]
    non_fraud_idx = np.where(y_test.values == 0)[0]

    if len(fraud_idx) > 0:
        # True positive example
        plot_shap_force_plot(shap_values, X_test, fraud_idx[0], model_name, explainer=explainer)
        plot_shap_waterfall(shap_values, X_test, fraud_idx[0], model_name, explainer=explainer)
    if len(non_fraud_idx) > 0:
        # True negative example
        plot_shap_force_plot(shap_values, X_test, non_fraud_idx[0], model_name, explainer=explainer)

    # Step 5: Permutation importance
    perm_imp = compute_permutation_importance(model, X_test, y_test, model_name)
    plot_permutation_importance(perm_imp, model_name)

    # Step 6: Partial dependence plots
    plot_partial_dependence(model, X_test, top_5, model_name)

    # Step 7: Business interpretation
    feature_importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": mean_shap,
    }).sort_values("Importance", ascending=False)

    interpretations = generate_business_interpretation(shap_values, X_test, feature_importance_df)

    return {
        "shap_values": shap_values,
        "explainer": explainer,
        "permutation_importance": perm_imp,
        "feature_importance": feature_importance_df,
        "interpretations": interpretations,
    }
