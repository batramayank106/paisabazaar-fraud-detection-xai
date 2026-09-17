"""
Exploratory Data Analysis module.
===================================
Reusable visualization and analysis functions for the fraud detection project.
Uses Plotly for interactive charts and Seaborn/Matplotlib for static plots.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings("ignore")

from src.config import (
    COLORS, SEABORN_PALETTE, PLOTLY_COLORWAY,
    PLOTLY_TEMPLATE, PLOT_CONFIG, FIGURES_DIR,
    RANDOM_SEED,
)
from src.utils import print_section_header, setup_logging

logger = setup_logging(__name__)

# Apply matplotlib style
plt.rcParams.update(PLOT_CONFIG)
sns.set_palette(SEABORN_PALETTE)


def save_figure(fig, name: str, save: bool = True) -> None:
    """Save a Plotly or Matplotlib figure to the outputs directory."""
    if not save:
        return
    path = FIGURES_DIR / f"{name}.png"
    try:
        if hasattr(fig, "write_image"):
            fig.write_image(str(path), width=1200, height=600, scale=2)
        else:
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not save {name}: {e}")
        plt.close("all")


# ============================================================
# 1. FRAUD DISTRIBUTION
# ============================================================

def plot_fraud_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Donut chart showing fraud vs non-fraud distribution."""
    counts = df["Is_Fraud"].value_counts()
    labels = ["Non-Fraud", "Fraud"]
    colors = [COLORS["non_fraud"], COLORS["fraud"]]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["Fraud Distribution (Count)", "Fraud Distribution (%)"],
    )

    fig.add_trace(go.Pie(
        labels=labels, values=counts.values,
        marker=dict(colors=colors),
        textinfo="label+value",
        hole=0.4,
    ), row=1, col=1)

    fig.add_trace(go.Pie(
        labels=labels, values=counts.values,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hole=0.4,
    ), row=1, col=2)

    fig.update_layout(
        title_text="Fraud Distribution Analysis",
        template=PLOTLY_TEMPLATE,
        colorway=PLOTLY_COLORWAY,
        showlegend=True,
        height=450,
    )
    fig.show()
    save_figure(fig, "fraud_distribution")


# ============================================================
# 2. TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

def plot_amount_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Distribution of financial features by fraud status."""
    features = [
        ("Annual_Income", "Annual Income"),
        ("Monthly_Inhand_Salary", "Monthly In-Hand Salary"),
        ("Outstanding_Debt", "Outstanding Debt"),
        ("Total_EMI_per_month", "Total EMI per Month"),
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f[1] for f in features])

    for idx, (col, title) in enumerate(features):
        row = idx // 2 + 1
        c = idx % 2 + 1

        for fraud_val, color, label in [(0, COLORS["non_fraud"], "Non-Fraud"),
                                         (1, COLORS["fraud"], "Fraud")]:
            subset = df[df["Is_Fraud"] == fraud_val][col].dropna()
            fig.add_trace(go.Histogram(
                x=subset, name=label,
                opacity=0.7, marker_color=color,
                nbinsx=40, showlegend=(idx == 0),
            ), row=row, col=c)

    fig.update_layout(
        title_text="Financial Feature Distributions by Fraud Status",
        template=PLOTLY_TEMPLATE, barmode="overlay", height=700,
    )
    fig.show()
    save_figure(fig, "amount_distribution")


def plot_amount_boxplots(df: pd.DataFrame, save: bool = True) -> None:
    """Boxplots and violin plots for key financial features."""
    features = ["Annual_Income", "Outstanding_Debt", "Credit_Utilization_Ratio",
                "Monthly_Balance", "Total_EMI_per_month", "Amount_invested_monthly"]
    features = [f for f in features if f in df.columns]

    fig = make_subplots(rows=2, cols=3, subplot_titles=features)

    for idx, col in enumerate(features):
        row = idx // 3 + 1
        c = idx % 3 + 1
        for fraud_val, color, label in [(0, COLORS["non_fraud"], "Non-Fraud"),
                                         (1, COLORS["fraud"], "Fraud")]:
            subset = df[df["Is_Fraud"] == fraud_val][col].dropna()
            fig.add_trace(go.Box(
                y=subset, name=label,
                marker_color=color, showlegend=(idx == 0),
            ), row=row, col=c)

    fig.update_layout(
        title_text="Box Plots: Feature Distributions by Fraud Status",
        template=PLOTLY_TEMPLATE, height=700,
    )
    fig.show()
    save_figure(fig, "amount_boxplots")


# ============================================================
# 3. CUSTOMER BEHAVIOR ANALYSIS
# ============================================================

def plot_age_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Age distribution by fraud status."""
    fig = px.histogram(
        df, x="Age", color="Is_Fraud",
        barmode="overlay", opacity=0.7,
        color_discrete_map={0: COLORS["non_fraud"], 1: COLORS["fraud"]},
        labels={"Is_Fraud": "Fraud Status", "0": "Non-Fraud", "1": "Fraud"},
        title="Age Distribution by Fraud Status",
        nbins=40,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    fig.show()
    save_figure(fig, "age_distribution")


def plot_occupation_fraud(df: pd.DataFrame, save: bool = True) -> None:
    """Fraud rate by occupation with bar chart."""
    occ_fraud = df.groupby("Occupation").agg(
        fraud_rate=("Is_Fraud", "mean"),
        count=("Is_Fraud", "count"),
        fraud_count=("Is_Fraud", "sum"),
    ).reset_index().sort_values("fraud_rate", ascending=True)

    fig = px.bar(
        occ_fraud, x="fraud_rate", y="Occupation",
        orientation="h", color="fraud_rate",
        color_continuous_scale=[COLORS["non_fraud"], COLORS["fraud"]],
        title="Fraud Rate by Occupation",
        labels={"fraud_rate": "Fraud Rate", "Occupation": ""},
        text="fraud_rate",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=550, coloraxis_showscale=False)
    fig.show()
    save_figure(fig, "occupation_fraud")


def plot_credit_score_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """Credit score distribution by fraud status."""
    ct = pd.crosstab(df["Credit_Score"], df["Is_Fraud"], normalize="index")

    fig = go.Figure()
    for fraud_val, color, label in [(0, COLORS["non_fraud"], "Non-Fraud"),
                                     (1, COLORS["fraud"], "Fraud")]:
        fig.add_trace(go.Bar(
            x=ct.index, y=ct[fraud_val],
            name=label, marker_color=color,
        ))

    fig.update_layout(
        barmode="stack", template=PLOTLY_TEMPLATE,
        title="Credit Score Distribution by Fraud Status",
        xaxis_title="Credit Score", yaxis_title="Proportion",
        height=450,
    )
    fig.show()
    save_figure(fig, "credit_score_distribution")


# ============================================================
# 4. PAYMENT BEHAVIOR ANALYSIS
# ============================================================

def plot_payment_behavior_sunburst(df: pd.DataFrame, save: bool = True) -> None:
    """Sunburst chart of Payment_Behaviour categories."""
    if "Spend_Level" not in df.columns or "Transaction_Value" not in df.columns:
        return

    ct = df.groupby(["Spend_Level", "Transaction_Value", "Is_Fraud"]).size().reset_index(name="Count")
    ct["Fraud_Label"] = ct["Is_Fraud"].map({0: "Non-Fraud", 1: "Fraud"})

    fig = px.sunburst(
        ct, path=["Spend_Level", "Transaction_Value", "Fraud_Label"],
        values="Count", color="Count",
        color_continuous_scale="RdYlBu_r",
        title="Payment Behavior: Spend Level -> Transaction Value -> Fraud",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=600)
    fig.show()
    save_figure(fig, "payment_behavior_sunburst")


def plot_payment_behavior_heatmap(df: pd.DataFrame, save: bool = True) -> None:
    """Heatmap of payment behavior categories vs fraud rate."""
    if "Payment_Behaviour" not in df.columns:
        return

    ct = df.groupby("Payment_Behaviour")["Is_Fraud"].agg(["mean", "count"]).reset_index()
    ct.columns = ["Payment_Behaviour", "Fraud_Rate", "Count"]

    fig = px.bar(
        ct, x="Payment_Behaviour", y="Fraud_Rate",
        color="Fraud_Rate",
        color_continuous_scale=[COLORS["non_fraud"], COLORS["fraud"]],
        title="Fraud Rate by Payment Behavior",
        labels={"Fraud_Rate": "Fraud Rate", "Payment_Behaviour": ""},
        text="Count",
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=500,
        xaxis_tickangle=-45, coloraxis_showscale=False,
    )
    fig.show()
    save_figure(fig, "payment_behavior_heatmap")


# ============================================================
# 5. TIME-BASED ANALYSIS
# ============================================================

def plot_monthly_fraud_trend(df: pd.DataFrame, save: bool = True) -> None:
    """Interactive timeline of fraud rate by month."""
    monthly = df.groupby("Month").agg(
        fraud_rate=("Is_Fraud", "mean"),
        total=("Is_Fraud", "count"),
        fraud_count=("Is_Fraud", "sum"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=monthly["Month"], y=monthly["fraud_rate"],
        name="Fraud Rate", mode="lines+markers",
        line=dict(color=COLORS["fraud"], width=3),
        marker=dict(size=10),
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=monthly["Month"], y=monthly["total"],
        name="Total Transactions", opacity=0.3,
        marker_color=COLORS["non_fraud"],
    ), secondary_y=True)

    fig.update_layout(
        title="Monthly Fraud Rate Trend",
        template=PLOTLY_TEMPLATE, height=450,
        xaxis_title="Month",
    )
    fig.update_yaxes(title_text="Fraud Rate", secondary_y=False)
    fig.update_yaxes(title_text="Total Transactions", secondary_y=True)
    fig.show()
    save_figure(fig, "monthly_fraud_trend")


def plot_credit_history_fraud(df: pd.DataFrame, save: bool = True) -> None:
    """Credit history age distribution by fraud status."""
    fig = px.violin(
        df, x="Is_Fraud", y="Credit_History_Age",
        color="Is_Fraud", box=True, points="outliers",
        color_discrete_map={0: COLORS["non_fraud"], 1: COLORS["fraud"]},
        title="Credit History Age by Fraud Status",
        labels={"Is_Fraud": "Fraud", "Credit_History_Age": "Credit History Age (months)"},
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=450)
    fig.show()
    save_figure(fig, "credit_history_fraud")


# ============================================================
# 6. RISK INDICATORS
# ============================================================

def plot_risk_indicators(df: pd.DataFrame, save: bool = True) -> None:
    """Comprehensive risk indicator plots."""
    features = [
        ("Num_of_Delayed_Payment", "Delayed Payments"),
        ("Num_Credit_Inquiries", "Credit Inquiries"),
        ("Credit_Utilization_Ratio", "Credit Utilization (%)"),
        ("Outstanding_Debt", "Outstanding Debt"),
    ]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f[1] for f in features])

    for idx, (col, title) in enumerate(features):
        row = idx // 2 + 1
        c = idx % 2 + 1
        for fraud_val, color, label in [(0, COLORS["non_fraud"], "Non-Fraud"),
                                         (1, COLORS["fraud"], "Fraud")]:
            subset = df[df["Is_Fraud"] == fraud_val][col].dropna()
            fig.add_trace(go.Violin(
                y=subset, name=label, box_visible=True,
                meanline_visible=True, opacity=0.6,
                line_color=color, showlegend=(idx == 0),
            ), row=row, col=c)

    fig.update_layout(
        title_text="Risk Indicator Distributions by Fraud Status",
        template=PLOTLY_TEMPLATE, height=700,
    )
    fig.show()
    save_figure(fig, "risk_indicators")


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

def plot_correlation_matrix(df: pd.DataFrame, save: bool = True) -> None:
    """Full correlation matrix heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    # Select key columns to avoid overcrowding
    key_cols = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
        "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Credit_Utilization_Ratio", "Outstanding_Debt",
        "Credit_History_Age", "Total_EMI_per_month",
        "Monthly_Balance", "Is_Fraud",
    ]
    key_cols = [c for c in key_cols if c in numeric_df.columns]
    corr = numeric_df[key_cols].corr()

    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix (Key Features)",
        aspect="auto",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=700, width=900)
    fig.show()
    save_figure(fig, "correlation_matrix")


def plot_fraud_correlations(df: pd.DataFrame, save: bool = True) -> None:
    """Bar chart of features most correlated with Is_Fraud."""
    numeric_df = df.select_dtypes(include=[np.number])
    fraud_corr = numeric_df.corr()["Is_Fraud"].drop("Is_Fraud").sort_values()

    colors = [COLORS["fraud"] if v > 0 else COLORS["non_fraud"] for v in fraud_corr.values]

    fig = go.Figure(go.Bar(
        x=fraud_corr.values, y=fraud_corr.index,
        orientation="h", marker_color=colors,
    ))
    fig.update_layout(
        title="Feature Correlations with Fraud Label",
        template=PLOTLY_TEMPLATE, height=600,
        xaxis_title="Pearson Correlation Coefficient",
    )
    fig.show()
    save_figure(fig, "fraud_correlations")


# ============================================================
# 8. HIGH-RISK SEGMENTS
# ============================================================

def plot_risk_segmentation(df: pd.DataFrame, save: bool = True) -> None:
    """Bubble chart of risk segments."""
    age_bins = pd.cut(df["Age"], bins=[0, 25, 35, 45, 56], labels=["18-25", "26-35", "36-45", "46-56"])
    seg = df.groupby([age_bins, "Credit_Score"]).agg(
        fraud_rate=("Is_Fraud", "mean"),
        count=("Is_Fraud", "count"),
    ).reset_index()

    fig = px.scatter(
        seg, x="Age", y="fraud_rate",
        size="count", color="Credit_Score",
        color_discrete_map={"Good": COLORS["good"], "Standard": COLORS["standard"], "Poor": COLORS["poor"]},
        title="Risk Segmentation: Age Group vs Credit Score",
        labels={"fraud_rate": "Fraud Rate", "Age": "Age Group", "count": "N"},
        size_max=60,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=500)
    fig.show()
    save_figure(fig, "risk_segmentation")


def plot_treemap_occupation(df: pd.DataFrame, save: bool = True) -> None:
    """Treemap of fraud by occupation."""
    occ_data = df.groupby(["Occupation", "Credit_Score"]).agg(
        count=("Is_Fraud", "count"),
        fraud_rate=("Is_Fraud", "mean"),
    ).reset_index()

    fig = px.treemap(
        occ_data, path=["Occupation", "Credit_Score"],
        values="count", color="fraud_rate",
        color_continuous_scale=[COLORS["non_fraud"], COLORS["fraud"]],
        title="Fraud Rate Treemap: Occupation -> Credit Score",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=600)
    fig.show()
    save_figure(fig, "treemap_occupation")


# ============================================================
# 9. BEHAVIOR BEFORE FRAUD (LONGITUDINAL)
# ============================================================

def plot_longitudinal_patterns(df: pd.DataFrame, save: bool = True) -> None:
    """Track customer behavior over months for fraud vs non-fraud."""
    if "Customer_ID" not in df.columns:
        return

    fraud_customers = df[df["Is_Fraud"] == 1]["Customer_ID"].unique()
    non_fraud_customers = df[df["Is_Fraud"] == 0]["Customer_ID"].unique()

    sample_fraud = np.random.choice(fraud_customers, min(50, len(fraud_customers)), replace=False)
    sample_non_fraud = np.random.choice(non_fraud_customers, min(50, len(non_fraud_customers)), replace=False)

    fraud_trajectories = df[df["Customer_ID"].isin(sample_fraud)].groupby("Month")[
        ["Num_of_Delayed_Payment", "Credit_Utilization_Ratio", "Outstanding_Debt"]
    ].mean()
    non_fraud_traj = df[df["Customer_ID"].isin(sample_non_fraud)].groupby("Month")[
        ["Num_of_Delayed_Payment", "Credit_Utilization_Ratio", "Outstanding_Debt"]
    ].mean()

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Delayed Payments", "Credit Utilization", "Outstanding Debt"])

    for col_idx, metric in enumerate(["Num_of_Delayed_Payment", "Credit_Utilization_Ratio", "Outstanding_Debt"]):
        fig.add_trace(go.Scatter(
            x=fraud_trajectories.index, y=fraud_trajectories[metric],
            name="Fraud", line=dict(color=COLORS["fraud"], width=2),
            showlegend=(col_idx == 0),
        ), row=1, col=col_idx + 1)
        fig.add_trace(go.Scatter(
            x=non_fraud_traj.index, y=non_fraud_traj[metric],
            name="Non-Fraud", line=dict(color=COLORS["non_fraud"], width=2),
            showlegend=(col_idx == 0),
        ), row=1, col=col_idx + 1)

    fig.update_layout(
        title="Behavioral Trajectories: Fraud vs Non-Fraud Customers",
        template=PLOTLY_TEMPLATE, height=450,
    )
    fig.show()
    save_figure(fig, "longitudinal_patterns")


# ============================================================
# 10. SCATTER PLOTS
# ============================================================

def plot_scatter_analysis(df: pd.DataFrame, save: bool = True) -> None:
    """Scatter plots of key feature pairs colored by fraud."""
    pairs = [
        ("Annual_Income", "Outstanding_Debt"),
        ("Credit_Utilization_Ratio", "Num_of_Delayed_Payment"),
        ("Monthly_Balance", "Total_EMI_per_month"),
        ("Credit_History_Age", "Credit_Utilization_Ratio"),
    ]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{x} vs {y}" for x, y in pairs])

    for idx, (x, y) in enumerate(pairs):
        row = idx // 2 + 1
        c = idx % 2 + 1
        for fraud_val, color, label in [(0, COLORS["non_fraud"], "Non-Fraud"),
                                         (1, COLORS["fraud"], "Fraud")]:
            subset = df[df["Is_Fraud"] == fraud_val]
            fig.add_trace(go.Scatter(
                x=subset[x], y=subset[y],
                mode="markers", name=label,
                marker=dict(color=color, size=3, opacity=0.4),
                showlegend=(idx == 0),
            ), row=row, col=c)

    fig.update_layout(
        title_text="Scatter Analysis: Key Feature Pairs",
        template=PLOTLY_TEMPLATE, height=700,
    )
    fig.show()
    save_figure(fig, "scatter_analysis")


# ============================================================
# 11. FEATURE IMPORTANCE PRE-MODELING
# ============================================================

def plot_mutual_information(df: pd.DataFrame, target: str = "Is_Fraud",
                            save: bool = True) -> pd.DataFrame:
    """Compute and plot mutual information scores."""
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    numeric_df = numeric_df.fillna(0)

    X = numeric_df.drop(columns=[target], errors="ignore")
    y = df[target]

    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "MI_Score": mi_scores,
    }).sort_values("MI_Score", ascending=True)

    fig = go.Figure(go.Bar(
        x=mi_df["MI_Score"], y=mi_df["Feature"],
        orientation="h",
        marker_color=px.colors.sequential.Reds[-len(mi_df):],
    ))
    fig.update_layout(
        title="Mutual Information Scores (Feature Relevance to Fraud)",
        template=PLOTLY_TEMPLATE, height=max(400, len(mi_df) * 25),
        xaxis_title="MI Score",
    )
    fig.show()
    save_figure(fig, "mutual_information")

    return mi_df.sort_values("MI_Score", ascending=False)


# ============================================================
# 12. PAIR PLOTS
# ============================================================

def plot_pairplot(df: pd.DataFrame, features: list = None,
                  sample_size: int = 2000, save: bool = True) -> None:
    """Seaborn pairplot for a subset of features."""
    if features is None:
        features = ["Annual_Income", "Outstanding_Debt",
                    "Credit_Utilization_Ratio", "Is_Fraud"]
    features = [f for f in features if f in df.columns]

    sample = df[features].sample(min(sample_size, len(df)), random_state=RANDOM_SEED)
    sample["Is_Fraud"] = sample["Is_Fraud"].map({0: "Non-Fraud", 1: "Fraud"})

    g = sns.pairplot(sample, hue="Is_Fraud", palette={
        "Non-Fraud": COLORS["non_fraud"], "Fraud": COLORS["fraud"]
    }, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
    g.fig.suptitle("Pair Plot: Key Features", y=1.02)

    save_figure(g.fig, "pairplot")
    if not save:
        plt.show()
