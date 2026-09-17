"""
Statistical Analysis module.
==============================
Comprehensive hypothesis testing, normality tests,
effect sizes, confidence intervals, and feature significance.

Author: Mayank Batra
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from src.config import COLORS, FIGURES_DIR, RANDOM_SEED
from src.utils import print_section_header, setup_logging

logger = setup_logging(__name__)


def save_figure(fig, name: str) -> None:
    """Save matplotlib figure."""
    path = FIGURES_DIR / f"{name}.png"
    try:
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close("all")


# ============================================================
# 1. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df: pd.DataFrame, target: str = "Is_Fraud") -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations with p-values.
    """
    print_section_header("Correlation Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target]

    results = []
    for col in numeric_cols:
        # Pearson
        pearson_r, pearson_p = stats.pearsonr(df[col].fillna(0), df[target])
        # Spearman
        spearman_r, spearman_p = stats.spearmanr(df[col].fillna(0), df[target])

        results.append({
            "Feature": col,
            "Pearson_r": round(pearson_r, 4),
            "Pearson_p": pearson_p,
            "Spearman_r": round(spearman_r, 4),
            "Spearman_p": spearman_p,
            "Pearson_Sig": "***" if pearson_p < 0.001 else ("**" if pearson_p < 0.01 else ("*" if pearson_p < 0.05 else "ns")),
            "Spearman_Sig": "***" if spearman_p < 0.001 else ("**" if spearman_p < 0.01 else ("*" if spearman_p < 0.05 else "ns")),
        })

    corr_df = pd.DataFrame(results).sort_values("Pearson_r", key=abs, ascending=False)
    print(corr_df.to_string(index=False))

    return corr_df


# ============================================================
# 2. HYPOTHESIS TESTING
# ============================================================

def hypothesis_testing(df: pd.DataFrame, target: str = "Is_Fraud") -> pd.DataFrame:
    """
    Perform t-tests and Mann-Whitney U tests for each numeric feature.
    Tests whether the feature distribution differs between fraud and non-fraud groups.
    """
    print_section_header("Hypothesis Testing: Fraud vs Non-Fraud")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target]

    results = []
    for col in numeric_cols:
        fraud_vals = df.loc[df[target] == 1, col].dropna()
        non_fraud_vals = df.loc[df[target] == 0, col].dropna()

        # Parametric: Welch's t-test (does not assume equal variance)
        t_stat, t_p = stats.ttest_ind(fraud_vals, non_fraud_vals, equal_var=False)

        # Non-parametric: Mann-Whitney U
        u_stat, u_p = stats.mannwhitneyu(fraud_vals, non_fraud_vals, alternative="two-sided")

        # Effect size: Cohen's d
        pooled_std = np.sqrt(
            ((len(fraud_vals) - 1) * fraud_vals.std()**2 +
             (len(non_fraud_vals) - 1) * non_fraud_vals.std()**2) /
            (len(fraud_vals) + len(non_fraud_vals) - 2)
        )
        cohens_d = (fraud_vals.mean() - non_fraud_vals.mean()) / pooled_std if pooled_std > 0 else 0

        results.append({
            "Feature": col,
            "t_stat": round(t_stat, 4),
            "t_p_value": t_p,
            "U_stat": round(u_stat, 0),
            "U_p_value": u_p,
            "Cohens_d": round(cohens_d, 4),
            "Effect_Size": (
                "Large" if abs(cohens_d) > 0.8 else
                ("Medium" if abs(cohens_d) > 0.5 else
                 ("Small" if abs(cohens_d) > 0.2 else "Negligible"))
            ),
            "Significant": t_p < 0.05,
        })

    results_df = pd.DataFrame(results).sort_values("t_p_value")

    # Apply Bonferroni correction
    n_tests = len(results_df)
    results_df["Bonferroni_Sig"] = results_df["t_p_value"] < (0.05 / n_tests)

    sig_count = results_df["Significant"].sum()
    bonf_count = results_df["Bonferroni_Sig"].sum()
    print(f"\nSignificant features (p < 0.05): {sig_count}/{n_tests}")
    print(f"Significant after Bonferroni correction: {bonf_count}/{n_tests}\n")

    print(results_df[["Feature", "t_stat", "t_p_value", "Cohens_d", "Effect_Size",
                       "Significant", "Bonferroni_Sig"]].to_string(index=False))

    return results_df


# ============================================================
# 3. ANOVA
# ============================================================

def anova_analysis(df: pd.DataFrame, target: str = "Is_Fraud") -> pd.DataFrame:
    """
    One-way ANOVA and Kruskal-Wallis tests.
    Groups: Credit_Score (Good, Standard, Poor)
    """
    print_section_header("ANOVA Analysis: Across Credit Score Groups")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in [target, "Credit_Score_Encoded"]]

    groups = df["Credit_Score"].unique()
    results = []

    for col in numeric_cols:
        group_data = [df.loc[df["Credit_Score"] == g, col].dropna() for g in groups]

        # One-way ANOVA
        f_stat, anova_p = stats.f_oneway(*group_data)

        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p = stats.kruskal(*group_data)

        results.append({
            "Feature": col,
            "F_stat": round(f_stat, 4),
            "ANOVA_p": anova_p,
            "H_stat": round(h_stat, 4),
            "KW_p": kw_p,
            "ANOVA_Sig": "***" if anova_p < 0.001 else ("**" if anova_p < 0.01 else ("*" if anova_p < 0.05 else "ns")),
        })

    results_df = pd.DataFrame(results).sort_values("ANOVA_p")
    print(results_df.to_string(index=False))

    return results_df


# ============================================================
# 4. CHI-SQUARE TEST
# ============================================================

def chi_square_analysis(df: pd.DataFrame, target: str = "Is_Fraud") -> pd.DataFrame:
    """
    Chi-Square test of independence for categorical features vs fraud.
    """
    print_section_header("Chi-Square Test: Categorical Features vs Fraud")

    cat_cols = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    results = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[target])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)

        # Cramer's V
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0

        results.append({
            "Feature": col,
            "Chi2": round(chi2, 4),
            "p_value": p,
            "DOF": dof,
            "Cramers_V": round(cramers_v, 4),
            "Association": (
                "Strong" if cramers_v > 0.5 else
                ("Moderate" if cramers_v > 0.3 else
                 ("Weak" if cramers_v > 0.1 else "Negligible"))
            ),
            "Significant": p < 0.05,
        })

    results_df = pd.DataFrame(results).sort_values("p_value")
    print(results_df.to_string(index=False))

    return results_df


# ============================================================
# 5. NORMALITY TESTS
# ============================================================

def normality_tests(df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    """
    Test normality of numeric features using D'Agostino-Pearson and
    Anderson-Darling tests.
    """
    print_section_header("Normality Tests")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ["Is_Fraud", "Customer_ID", "ID", "Month"]]

    results = []
    for col in numeric_cols:
        sample = df[col].dropna()
        if len(sample) > sample_size:
            sample = sample.sample(sample_size, random_state=RANDOM_SEED)

        # D'Agostino-Pearson
        if len(sample) >= 20:
            dp_stat, dp_p = stats.normaltest(sample)
        else:
            dp_stat, dp_p = np.nan, np.nan

        # Anderson-Darling
        ad_result = stats.anderson(sample, dist="norm")
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% significance level

        results.append({
            "Feature": col,
            "DAgostino_stat": round(dp_stat, 4) if not np.isnan(dp_stat) else "N/A",
            "DAgostino_p": round(dp_p, 6) if not np.isnan(dp_p) else "N/A",
            "AD_stat": round(ad_stat, 4),
            "AD_Critical_5pct": round(ad_critical, 4),
            "Normal": "No" if (not np.isnan(dp_p) and dp_p < 0.05) or ad_stat > ad_critical else "Yes",
        })

    results_df = pd.DataFrame(results)
    non_normal = results_df[results_df["Normal"] == "No"].shape[0]
    print(f"\nNon-normal features: {non_normal}/{len(results_df)}")
    print(f"(Most financial data is non-normal, which is expected)\n")
    print(results_df.to_string(index=False))

    return results_df


# ============================================================
# 6. CONFIDENCE INTERVALS
# ============================================================

def confidence_intervals(df: pd.DataFrame, target: str = "Is_Fraud",
                         confidence: float = 0.95) -> pd.DataFrame:
    """
    Compute confidence intervals for fraud rate and key metrics.
    """
    print_section_header(f"{int(confidence*100)}% Confidence Intervals")

    fraud_rate = df[target].mean()
    n = len(df)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(fraud_rate * (1 - fraud_rate) / n)
    ci_lower = fraud_rate - z * se
    ci_upper = fraud_rate + z * se

    print(f"\n  Overall Fraud Rate: {fraud_rate:.4f}")
    print(f"  {int(confidence*100)}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Margin of Error: +/- {z * se:.4f}")

    # CI for mean of key features by group
    key_features = ["Annual_Income", "Outstanding_Debt", "Credit_Utilization_Ratio"]
    key_features = [f for f in key_features if f in df.columns]

    results = []
    for col in key_features:
        for fraud_val, label in [(0, "Non-Fraud"), (1, "Fraud")]:
            subset = df.loc[df[target] == fraud_val, col].dropna()
            mean = subset.mean()
            se = subset.std() / np.sqrt(len(subset))
            ci_l = mean - z * se
            ci_u = mean + z * se
            results.append({
                "Group": label,
                "Feature": col,
                "Mean": round(mean, 4),
                "CI_Lower": round(ci_l, 4),
                "CI_Upper": round(ci_u, 4),
            })

    ci_df = pd.DataFrame(results)
    print(f"\n{ci_df.to_string(index=False)}")

    return ci_df


# ============================================================
# 7. EFFECT SIZE SUMMARY
# ============================================================

def effect_size_summary(df: pd.DataFrame, target: str = "Is_Fraud") -> pd.DataFrame:
    """
    Consolidated effect size analysis for all features.
    """
    print_section_header("Effect Size Summary")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != target]

    results = []
    for col in numeric_cols:
        fraud_vals = df.loc[df[target] == 1, col].dropna()
        non_fraud_vals = df.loc[df[target] == 0, col].dropna()

        pooled_std = np.sqrt(
            ((len(fraud_vals) - 1) * fraud_vals.std()**2 +
             (len(non_fraud_vals) - 1) * non_fraud_vals.std()**2) /
            (len(fraud_vals) + len(non_fraud_vals) - 2)
        )
        cohens_d = (fraud_vals.mean() - non_fraud_vals.mean()) / pooled_std if pooled_std > 0 else 0

        # Rank-biserial correlation (effect size for Mann-Whitney)
        u_stat, _ = stats.mannwhitneyu(fraud_vals, non_fraud_vals, alternative="two-sided")
        r = 1 - (2 * u_stat) / (len(fraud_vals) * len(non_fraud_vals))

        results.append({
            "Feature": col,
            "Cohens_d": round(cohens_d, 4),
            "Rank_Biserial_r": round(r, 4),
            "Practical_Significance": (
                "High" if abs(cohens_d) > 0.8 else
                ("Medium" if abs(cohens_d) > 0.5 else
                 ("Low" if abs(cohens_d) > 0.2 else "Negligible"))
            ),
        })

    results_df = pd.DataFrame(results).sort_values("Cohens_d", key=abs, ascending=False)
    print(results_df.to_string(index=False))

    return results_df


# ============================================================
# 8. FEATURE SIGNIFICANCE SUMMARY
# ============================================================

def feature_significance_summary(
    corr_results: pd.DataFrame,
    hypothesis_results: pd.DataFrame,
    chi2_results: pd.DataFrame,
    effect_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Consolidate all statistical test results into a single summary table.
    """
    print_section_header("Feature Significance Summary")

    # Merge numeric results
    merged = hypothesis_results[["Feature", "t_p_value", "Cohens_d", "Significant"]].merge(
        effect_results[["Feature", "Rank_Biserial_r", "Practical_Significance"]],
        on="Feature", how="outer"
    )

    # Add categorical results
    chi2_summary = chi2_results[["Feature", "Chi2", "p_value", "Cramers_V", "Significant"]].copy()
    chi2_summary.columns = ["Feature", "Chi2", "Chi2_p_value", "Cramers_V", "Chi2_Significant"]

    merged = pd.concat([merged, chi2_summary], ignore_index=True)

    # Add correlation
    corr_summary = corr_results[["Feature", "Pearson_r", "Pearson_p"]].copy()
    merged = merged.merge(corr_summary, on="Feature", how="left")

    # Sort by p-value
    merged = merged.sort_values("t_p_value", ascending=True, na_position="last")

    # Final significance verdict
    merged["Overall_Significant"] = merged["Significant"].fillna(merged.get("Chi2_Significant", False))

    sig_features = merged[merged["Overall_Significant"] == True]["Feature"].tolist()
    print(f"\nStatistically significant features ({len(sig_features)}):")
    for f in sig_features:
        print(f"  - {f}")

    return merged
