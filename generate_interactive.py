"""
Generate Standalone Interactive Plotly HTML Files
===================================================
Runs the full pipeline and creates self-contained interactive HTML
dashboards in outputs/interactive/. Each file opens in any browser.

Author: Mayank Batra
"""

import os
import sys
from pathlib import Path

# Portable project root (works on any clone location)
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import plotly.io as pio
pio.renderers.default = "png"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

go.Figure.show = lambda self, *a, **kw: None

from src.config import RANDOM_SEED, PLOTLY_TEMPLATE
from src.utils import set_all_seeds
from src.data_loader import load_data
from src.data_cleaning import run_cleaning_pipeline
from src.feature_engineering import run_feature_engineering
from src.modeling import run_full_ml_pipeline

set_all_seeds(RANDOM_SEED)

INTERACTIVE_DIR = PROJECT_ROOT / "outputs" / "interactive"
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "Logistic Regression": "#636EFA",
    "Decision Tree": "#EF553B",
    "Random Forest": "#00CC96",
    "XGBoost": "#AB63FA",
    "LightGBM": "#FFA15A",
    "CatBoost": "#19D3F3",
}

TEMPLATE = "plotly_white"


def save(fig, name):
    path = INTERACTIVE_DIR / name
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  [OK] {name}")


# ============================================================
# 1. Run full pipeline
# ============================================================
print("=" * 60)
print("  RUNNING FULL PIPELINE")
print("=" * 60)

df_raw = load_data()
df_clean = run_cleaning_pipeline(df_raw)
df_feat = run_feature_engineering(df_clean)
ml = run_full_ml_pipeline(df_feat, target="Is_Fraud", imbalance_method="class_weights")

models = ml["models"]
metrics_df = ml["metrics"]
data = ml["data"]
best_name = ml["best_model_name"]
best_model = models[best_name]

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
X_test_unscaled = data["X_test_unscaled"]
feature_names = data["feature_names"]

print(f"\nBest model: {best_name}")
print(f"Models available: {list(models.keys())}")

# ============================================================
# Compute SHAP values for LightGBM
# ============================================================
print("\n" + "=" * 60)
print("  COMPUTING SHAP VALUES FOR LightGBM")
print("=" * 60)

import shap

lgb_model = models["LightGBM"]
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_shap, index=X_test.columns).sort_values(ascending=False)
top_5_features = shap_importance.head(5).index.tolist()

print(f"  SHAP values computed: {shap_values.shape}")
print(f"  Top 5 features: {top_5_features}")

# Permutation importance for LightGBM
print("\n  Computing permutation importance...")
from sklearn.inspection import permutation_importance as perm_imp

pi_result = perm_imp(
    lgb_model, X_test, y_test,
    n_repeats=10, random_state=RANDOM_SEED,
    scoring="roc_auc", n_jobs=-1,
)
perm_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": pi_result.importances_mean,
    "Importance_Std": pi_result.importances_std,
}).sort_values("Importance_Mean", ascending=False)

print("  Permutation importance computed.")

# ============================================================
# Generate all interactive HTML files
# ============================================================
print("\n" + "=" * 60)
print("  GENERATING INTERACTIVE HTML FILES")
print("=" * 60)

# ----------------------------------------------------------
# 1. model_comparison.html
# ----------------------------------------------------------
print("\n[1/11] Model Comparison...")
key_metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=key_metrics,
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)
for idx, metric in enumerate(key_metrics):
    row = idx // 3 + 1
    col = idx % 3 + 1
    sorted_df = metrics_df.sort_values(metric, ascending=True)
    for _, mrow in sorted_df.iterrows():
        fig.add_trace(go.Bar(
            x=[mrow[metric]], y=[mrow["Model"]],
            orientation="h", name=mrow["Model"],
            marker_color=MODEL_COLORS.get(mrow["Model"], "#999"),
            showlegend=(idx == 0),
            hovertemplate=f"<b>{mrow['Model']}</b><br>{metric}: {mrow[metric]:.4f}<extra></extra>",
            text=[f"{mrow[metric]:.4f}"],
            textposition="outside",
            textfont=dict(size=9),
        ), row=row, col=col)

fig.update_layout(
    title_text="Model Comparison Across All Metrics",
    title_font_size=18,
    template=TEMPLATE,
    height=700,
    width=1100,
    showlegend=True,
    legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
    barmode="group",
)
for i in range(1, 4):
    fig.update_xaxes(range=[0, 1.15], row=1, col=i)
for i in range(1, 4):
    fig.update_xaxes(range=[0, 1.15], row=2, col=i)
save(fig, "model_comparison.html")

# ----------------------------------------------------------
# 2. roc_curves.html
# ----------------------------------------------------------
print("\n[2/11] ROC Curves...")
from sklearn.metrics import roc_curve as sk_roc, roc_auc_score

fig = go.Figure()
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = sk_roc(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    step = max(1, len(fpr) // 200)
    fig.add_trace(go.Scatter(
        x=fpr[::step], y=tpr[::step],
        name=f"{name} (AUC={auc:.4f})",
        mode="lines",
        line=dict(width=2.5, color=MODEL_COLORS.get(name, "#999")),
        hovertemplate="FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra>" + name + "</extra>",
    ))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], name="Random (AUC=0.5000)",
    mode="lines", line=dict(dash="dash", color="gray", width=1.5),
))
fig.update_layout(
    title="ROC Curves — All Models",
    title_font_size=18,
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    template=TEMPLATE, width=1000, height=600,
    legend=dict(x=0.55, y=0.05, bgcolor="rgba(255,255,255,0.8)"),
    xaxis=dict(range=[-0.01, 1.01]),
    yaxis=dict(range=[-0.01, 1.01]),
)
save(fig, "roc_curves.html")

# ----------------------------------------------------------
# 3. pr_curves.html
# ----------------------------------------------------------
print("\n[3/11] Precision-Recall Curves...")
from sklearn.metrics import precision_recall_curve as sk_pr, average_precision_score

fig = go.Figure()
baseline = y_test.mean()
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = sk_pr(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    step = max(1, len(precision) // 200)
    fig.add_trace(go.Scatter(
        x=recall[::step], y=precision[::step],
        name=f"{name} (AP={ap:.4f})",
        mode="lines",
        line=dict(width=2.5, color=MODEL_COLORS.get(name, "#999")),
        hovertemplate="Recall: %{x:.4f}<br>Precision: %{y:.4f}<extra>" + name + "</extra>",
    ))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[baseline, baseline], name=f"Baseline ({baseline:.4f})",
    mode="lines", line=dict(dash="dash", color="gray", width=1.5),
))
fig.update_layout(
    title="Precision-Recall Curves — All Models",
    title_font_size=18,
    xaxis_title="Recall",
    yaxis_title="Precision",
    template=TEMPLATE, width=1000, height=600,
    legend=dict(x=0.55, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
    xaxis=dict(range=[-0.01, 1.01]),
    yaxis=dict(range=[-0.01, 1.05]),
)
save(fig, "pr_curves.html")

# ----------------------------------------------------------
# 4. confusion_matrices.html
# ----------------------------------------------------------
print("\n[4/11] Confusion Matrices...")
from sklearn.metrics import confusion_matrix

model_names = list(models.keys())
n_models = len(model_names)
cols = 3
rows_count = (n_models + cols - 1) // cols

fig = make_subplots(
    rows=rows_count, cols=cols,
    subplot_titles=model_names,
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)

for idx, name in enumerate(model_names):
    r = idx // cols + 1
    c = idx % cols + 1
    y_pred = models[name].predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Non-Fraud", "Fraud"]

    text = [[f"TN={cm[0][0]}", f"FP={cm[0][1]}"],
            [f"FN={cm[1][0]}", f"TP={cm[1][1]}"]]

    fig.add_trace(go.Heatmap(
        z=cm,
        x=labels, y=labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13},
        colorscale="Blues",
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra>" + name + "</extra>",
    ), row=r, col=c)

fig.update_layout(
    title_text="Confusion Matrices — All Models",
    title_font_size=18,
    template=TEMPLATE,
    width=1100,
    height=450 * rows_count,
    showlegend=False,
)
for idx in range(n_models):
    r = idx // cols + 1
    c = idx % cols + 1
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Non-Fraud", "Fraud"], row=r, col=c)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Non-Fraud", "Fraud"], autorange="reversed", row=r, col=c)

save(fig, "confusion_matrices.html")

# ----------------------------------------------------------
# 5. shap_summary.html
# ----------------------------------------------------------
print("\n[5/11] SHAP Summary (LightGBM)...")
top_n = 20
top_feats = shap_importance.head(top_n).index.tolist()
shap_sub = shap_values[:, [X_test.columns.get_loc(f) for f in top_feats]]
X_sub = X_test[top_feats]

fig = go.Figure()
for feat in reversed(top_feats):
    feat_idx = list(X_test.columns).index(feat)
    feat_vals = X_sub[feat].values
    shap_vals = shap_sub[:, top_feats.index(feat)]
    fig.add_trace(go.Scatter(
        x=shap_vals, y=[feat] * len(shap_vals),
        mode="markers",
        marker=dict(
            size=4,
            color=feat_vals,
            colorscale="RdBu_r",
            colorbar=dict(title="Feature<br>Value") if feat == top_feats[0] else None,
            showscale=(feat == top_feats[0]),
            opacity=0.6,
        ),
        name=feat,
        showlegend=False,
        hovertemplate=f"<b>{feat}</b><br>SHAP: %{{x:.4f}}<br>Value: %{{marker.color:.4f}}<extra></extra>",
    ))

fig.update_layout(
    title="SHAP Summary Plot — LightGBM (Fraud Class)",
    title_font_size=18,
    xaxis_title="SHAP Value (positive = pushes toward fraud)",
    template=TEMPLATE,
    width=1100,
    height=700,
    yaxis=dict(
        tickfont=dict(size=11),
    ),
)
save(fig, "shap_summary.html")

# ----------------------------------------------------------
# 6. shap_dependence.html
# ----------------------------------------------------------
print("\n[6/11] SHAP Dependence (LightGBM)...")
interaction_feat = shap_importance.index[min(5, len(shap_importance) - 1)]

fig = go.Figure()
dropdown_buttons = []

for i, feat in enumerate(top_5_features):
    feat_idx = list(X_test.columns).index(feat)
    feat_vals = X_test[feat].values
    shap_vals = shap_values[:, feat_idx]

    color_feat_idx = feat_idx
    for fi in range(len(X_test.columns)):
        if fi != feat_idx:
            color_feat_idx = fi
            break
    color_feat_name = X_test.columns[color_feat_idx]
    color_vals = X_test.iloc[:, color_feat_idx].values

    step = max(1, len(feat_vals) // 500)
    trace = go.Scatter(
        x=feat_vals[::step], y=shap_vals[::step],
        mode="markers",
        marker=dict(
            size=5,
            color=color_vals[::step],
            colorscale="Viridis",
            colorbar=dict(title=color_feat_name),
            opacity=0.6,
        ),
        name=feat,
        visible=(i == 0),
        hovertemplate=f"<b>{feat}</b><br>Value: %{{x:.4f}}<br>SHAP: %{{y:.4f}}<br>{color_feat_name}: %{{marker.color:.4f}}<extra></extra>",
    )
    fig.add_trace(trace)

    dropdown_buttons.append(dict(
        label=feat,
        method="update",
        args=[{"visible": [j == i for j in range(len(top_5_features))]},
              {"title": f"SHAP Dependence Plot — {feat} (LightGBM)"}],
    ))

fig.update_layout(
    title=f"SHAP Dependence Plot — {top_5_features[0]} (LightGBM)",
    title_font_size=18,
    xaxis_title="Feature Value",
    yaxis_title="SHAP Value",
    template=TEMPLATE,
    width=1000,
    height=600,
    updatemenus=[dict(
        type="dropdown",
        direction="down",
        x=0.0, y=1.18,
        xanchor="left", yanchor="top",
        buttons=dropdown_buttons,
        bgcolor="white",
        bordercolor="#ccc",
    )],
    annotations=[dict(
        text="Select feature:", showarrow=False,
        x=0.0, y=1.13, xref="paper", yref="paper",
        font=dict(size=13),
    )],
)
save(fig, "shap_dependence.html")

# ----------------------------------------------------------
# 7. threshold_analysis.html
# ----------------------------------------------------------
print("\n[7/11] Threshold Analysis (LightGBM)...")
from sklearn.metrics import precision_recall_fscore_support

y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
thresholds_arr = np.arange(0.05, 0.95, 0.01)
precisions, recalls, f1s = [], [], []
for t in thresholds_arr:
    yp = (y_prob_lgb >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_test, yp, average="binary", zero_division=0)
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)

best_f1_idx = np.argmax(f1s)
best_threshold = thresholds_arr[best_f1_idx]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=thresholds_arr, y=precisions, name="Precision",
    mode="lines", line=dict(color="#636EFA", width=2.5),
    hovertemplate="Threshold: %{x:.3f}<br>Precision: %{y:.4f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=thresholds_arr, y=recalls, name="Recall",
    mode="lines", line=dict(color="#EF553B", width=2.5),
    hovertemplate="Threshold: %{x:.3f}<br>Recall: %{y:.4f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=thresholds_arr, y=f1s, name="F1 Score",
    mode="lines", line=dict(color="#00CC96", width=2.5),
    hovertemplate="Threshold: %{x:.3f}<br>F1: %{y:.4f}<extra></extra>",
))

fig.add_vline(
    x=best_threshold, line_dash="dash", line_color="gray", line_width=1.5,
    annotation_text=f"Best F1 = {f1s[best_f1_idx]:.4f} @ {best_threshold:.2f}",
    annotation_position="top right",
    annotation_font_size=12,
)

fig.add_annotation(
    x=best_threshold, y=f1s[best_f1_idx],
    text=f"Optimal: {best_threshold:.2f}",
    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
    ax=50, ay=-30, font=dict(size=12, color="#333"),
)

fig.update_layout(
    title=f"Threshold Analysis — {best_name}",
    title_font_size=18,
    xaxis_title="Classification Threshold",
    yaxis_title="Score",
    template=TEMPLATE, width=1000, height=550,
    legend=dict(x=0.02, y=0.02),
    xaxis=dict(range=[0.04, 0.96]),
    yaxis=dict(range=[0, 1.05]),
)
save(fig, "threshold_analysis.html")

# ----------------------------------------------------------
# 8. lift_gain.html
# ----------------------------------------------------------
print("\n[8/11] Lift & Gain Curves...")

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Lift Curve", "Gain Curve"],
    horizontal_spacing=0.1,
)

for name, model in models.items():
    yp = model.predict_proba(X_test)[:, 1]
    sorted_idx = np.argsort(-yp)
    y_sorted = y_test.values[sorted_idx]
    n = len(y_sorted)
    fractions = np.arange(1, n + 1) / n
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    lift = cum_pos / fractions

    step = max(1, len(fractions) // 300)
    fig.add_trace(go.Scatter(
        x=fractions[::step], y=lift[::step], name=name,
        mode="lines",
        line=dict(width=2, color=MODEL_COLORS.get(name, "#999")),
        showlegend=True,
        hovertemplate=f"<b>{name}</b><br>% Pop: %{{x:.3f}}<br>Lift: %{{y:.3f}}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fractions[::step], y=cum_pos[::step], name=name,
        mode="lines",
        line=dict(width=2, color=MODEL_COLORS.get(name, "#999")),
        showlegend=False,
        hovertemplate=f"<b>{name}</b><br>% Pop: %{{x:.3f}}<br>Gain: %{{y:.3f}}<extra></extra>",
    ), row=1, col=2)

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
    title="Lift and Gain Curves — All Models",
    title_font_size=18,
    template=TEMPLATE, width=1100, height=500,
    legend=dict(x=0.5, y=-0.12, orientation="h", xanchor="center"),
)
fig.update_xaxes(title_text="% of Population", row=1, col=1)
fig.update_xaxes(title_text="% of Population", row=1, col=2)
fig.update_yaxes(title_text="Lift", row=1, col=1)
fig.update_yaxes(title_text="% of Positive Cases Captured", row=1, col=2)
save(fig, "lift_gain.html")

# ----------------------------------------------------------
# 9. calibration.html
# ----------------------------------------------------------
print("\n[9/11] Calibration Curves...")

fig = go.Figure()
n_bins = 15
bin_edges = np.linspace(0, 1, n_bins + 1)

for name, model in models.items():
    yp = model.predict_proba(X_test)[:, 1]
    centers, means = [], []
    for i in range(n_bins):
        mask = (yp >= bin_edges[i]) & (yp < bin_edges[i + 1])
        if mask.sum() > 0:
            centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            means.append(y_test.values[mask].mean())
    fig.add_trace(go.Scatter(
        x=centers, y=means, name=name,
        mode="lines+markers",
        line=dict(width=2, color=MODEL_COLORS.get(name, "#999")),
        marker=dict(size=6),
        hovertemplate=f"<b>{name}</b><br>Predicted: %{{x:.4f}}<br>Actual: %{{y:.4f}}<extra></extra>",
    ))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], name="Perfectly Calibrated",
    mode="lines", line=dict(dash="dash", color="gray", width=1.5),
))
fig.update_layout(
    title="Calibration Curves — All Models",
    title_font_size=18,
    xaxis_title="Mean Predicted Probability",
    yaxis_title="Fraction of Positives",
    template=TEMPLATE, width=1000, height=600,
    legend=dict(x=0.02, y=0.98),
    xaxis=dict(range=[-0.02, 1.02]),
    yaxis=dict(range=[-0.02, 1.02]),
)
save(fig, "calibration.html")

# ----------------------------------------------------------
# 10. feature_importance.html
# ----------------------------------------------------------
print("\n[10/11] Feature Importance (SHAP vs Permutation)...")

top_n_imp = 15
shap_top = shap_importance.head(top_n_imp).sort_values(ascending=True)
perm_top = perm_importance.head(top_n_imp).sort_values("Importance_Mean", ascending=True)

shap_norm = shap_top / shap_top.max() if shap_top.max() > 0 else shap_top
perm_vals = perm_top["Importance_Mean"]
perm_norm_vals = perm_vals / perm_vals.max() if perm_vals.max() > 0 else perm_vals

common_feats = list(set(shap_top.index) | set(perm_top.index))[:top_n_imp]
common_feats_sorted = sorted(common_feats, key=lambda f: shap_importance.get(f, 0), reverse=True)[:top_n_imp]
common_feats_sorted.reverse()

fig = make_subplots(rows=1, cols=2, subplot_titles=["SHAP Importance (LightGBM)", "Permutation Importance (LightGBM)"],
                    horizontal_spacing=0.15)

for feat in common_feats_sorted:
    sv = shap_importance.get(feat, 0)
    fig.add_trace(go.Bar(
        y=[feat], x=[sv], orientation="h", name="SHAP",
        marker_color="#FFA15A", showlegend=False,
        hovertemplate=f"<b>{feat}</b><br>Mean |SHAP|: {sv:.4f}<extra></extra>",
    ), row=1, col=1)

for feat in common_feats_sorted:
    prow = perm_importance[perm_importance["Feature"] == feat]
    if len(prow) > 0:
        iv = prow["Importance_Mean"].values[0]
        fig.add_trace(go.Bar(
            y=[feat], x=[iv], orientation="h", name="Permutation",
            marker_color="#19D3F3", showlegend=False,
            hovertemplate=f"<b>{feat}</b><br>Mean Importance: {iv:.4f}<extra></extra>",
        ), row=1, col=2)

fig.update_layout(
    title="Feature Importance Comparison — SHAP vs Permutation",
    title_font_size=18,
    template=TEMPLATE, width=1100, height=650,
    barmode="group",
)
fig.update_xaxes(title_text="Mean |SHAP Value|", row=1, col=1)
fig.update_xaxes(title_text="Mean Decrease in ROC-AUC", row=1, col=2)
save(fig, "feature_importance.html")

# ----------------------------------------------------------
# 11. prediction_explorer.html
# ----------------------------------------------------------
print("\n[11/11] Prediction Explorer...")

y_prob_best = best_model.predict_proba(X_test)[:, 1]
y_pred_best = best_model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_best,
    "Probability": y_prob_best,
}, index=X_test.index)

def classify(row):
    if row["Actual"] == 1 and row["Predicted"] == 1:
        return "TP"
    elif row["Actual"] == 0 and row["Predicted"] == 0:
        return "TN"
    elif row["Actual"] == 0 and row["Predicted"] == 1:
        return "FP"
    else:
        return "FN"

results["Result"] = results.apply(classify, axis=1)

# Stratified sample of 16 records
sample_records = []
for res_type in ["TP", "TN", "FP", "FN"]:
    subset = results[results["Result"] == res_type]
    n_pick = min(4, len(subset))
    if n_pick > 0:
        sample_records.append(subset.sample(n_pick, random_state=RANDOM_SEED))

if sample_records:
    sample = pd.concat(sample_records).sort_index()
else:
    sample = results.sample(16, random_state=RANDOM_SEED)

# Feature columns for display
display_cols = ["Age", "Annual_Income", "Outstanding_Debt",
                "Credit_Utilization_Ratio", "Num_of_Delayed_Payment"]
display_cols = [c for c in display_cols if c in X_test_unscaled.columns]
feat_display = X_test_unscaled.loc[sample.index, display_cols]

color_map = {"TP": "#2ecc71", "TN": "#3498db", "FP": "#e74c3c", "FN": "#f39c12"}

records = []
for idx_num, (orig_idx, row) in enumerate(sample.iterrows(), 1):
    actual_str = "Fraud" if row["Actual"] == 1 else "Non-Fraud"
    pred_str = "Fraud" if row["Predicted"] == 1 else "Non-Fraud"
    result = row["Result"]
    fv = feat_display.loc[orig_idx]
    age_val = fv.get("Age", "N/A") if "Age" in fv.index else "N/A"
    income_val = fv.get("Annual_Income", 0) if "Annual_Income" in fv.index else 0
    debt_val = fv.get("Outstanding_Debt", 0) if "Outstanding_Debt" in fv.index else 0
    util_val = fv.get("Credit_Utilization_Ratio", 0) if "Credit_Utilization_Ratio" in fv.index else 0

    records.append([
        idx_num, actual_str, pred_str,
        f"{row['Probability']:.4f}", result,
        str(age_val), f"${income_val:,.0f}", f"${debt_val:,.2f}",
        f"{util_val:.1f}%",
    ])

col_names = ["Record#", "Actual", "Predicted", "Probability", "Result",
             "Age", "Income", "Debt", "Utilization"]

header_colors = ["#2c3e50"] * len(col_names)
cell_colors = [header_colors]
for i, col in enumerate(col_names):
    col_colors = ["rgba(240,240,240,0.9)"] * len(records)
    if col == "Result":
        for j, rec in enumerate(records):
            c = color_map.get(rec[4], "#999")
            col_colors[j] = c
    elif col == "Actual":
        for j, rec in enumerate(records):
            col_colors[j] = "#ffcccc" if rec[1] == "Fraud" else "#ccffcc"
    elif col == "Predicted":
        for j, rec in enumerate(records):
            col_colors[j] = "#ffcccc" if rec[2] == "Fraud" else "#ccffcc"
    cell_colors.append(col_colors)

# Build customdata for hover
hover_texts = []
for j, rec in enumerate(records):
    orig_idx = sample.index[j]
    fv = feat_display.loc[orig_idx]
    lines = [f"<b>Record #{rec[0]}</b>"]
    lines.append(f"Actual: {rec[1]} | Predicted: {rec[2]}")
    lines.append(f"Probability: {rec[3]} | Result: <b>{rec[4]}</b>")
    lines.append("---")
    for c in display_cols:
        v = fv.get(c, "N/A") if hasattr(fv, "get") else "N/A"
        if isinstance(v, (int, float, np.integer, np.floating)):
            lines.append(f"{c}: {v:,.2f}" if abs(v) > 100 else f"{c}: {v:.4f}")
        else:
            lines.append(f"{c}: {v}")
    hover_texts.append("<br>".join(lines))

transposed = list(zip(*records))
transposed = [list(x) for x in transposed]

fig = go.Figure()
fig.add_trace(go.Table(
    header=dict(
        values=[f"<b>{c}</b>" for c in col_names],
        fill_color="#2c3e50",
        font=dict(color="white", size=13),
        align="center",
        height=35,
    ),
    cells=dict(
        values=transposed,
        fill_color=cell_colors,
        font=dict(size=12),
        align="center",
        height=30,
    ),
    columnwidth=[0.6, 0.9, 0.9, 0.9, 0.6, 0.5, 1.0, 1.0, 0.9],
))

fig.update_layout(
    title=f"Prediction Explorer — {best_name} (16 Stratified Examples)",
    title_font_size=18,
    template=TEMPLATE,
    width=1100,
    height=650,
    margin=dict(l=20, r=20, t=60, b=20),
)

# Add color legend annotation
fig.add_annotation(
    text="<b>Color Legend:</b>   <span style='color:#2ecc71'>TP (True Positive)</span>   "
         "<span style='color:#3498db'>TN (True Negative)</span>   "
         "<span style='color:#e74c3c'>FP (False Positive)</span>   "
         "<span style='color:#f39c12'>FN (False Negative)</span>",
    xref="paper", yref="paper",
    x=0.5, y=-0.02, showarrow=False,
    font=dict(size=12),
    xanchor="center",
)

save(fig, "prediction_explorer.html")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("  ALL 11 INTERACTIVE HTML FILES GENERATED")
print("=" * 60)
print(f"\nOutput directory: {INTERACTIVE_DIR}")
print("\nFiles:")
for fname in sorted(os.listdir(INTERACTIVE_DIR)):
    fpath = os.path.join(INTERACTIVE_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  - {fname:35s} ({size_kb:.1f} KB)")
print("\nOpen any file in a web browser to interact with the charts.")
