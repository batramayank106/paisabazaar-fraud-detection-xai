# Business Questions Answered

## 1. What patterns distinguish high-risk credit profiles from low-risk ones?
**Method:** EDA, statistical tests, SHAP feature importance
**Finding:** Credit utilization ratio, outstanding debt, and delayed payment count are the strongest discriminators. Customers with utilization above 70% and more than 17 delayed payments are overwhelmingly in the high-risk segment.

## 2. Which features most strongly predict credit risk?
**Method:** SHAP values, permutation importance
**Finding:** Top 5 features by importance:
1. Credit utilization ratio
2. Outstanding debt
3. Number of delayed payments
4. Interest rate
5. Payment delay risk (delayed payments / credit history age)

## 3. Is there a relationship between payment behavior and credit risk?
**Method:** Chi-square test, cross-tabulation
**Finding:** Yes. Customers who consistently pay only the minimum amount show significantly higher fraud rates (chi-square p < 0.001). Payment behavior is a moderate but statistically significant predictor.

## 4. Do different occupations show different risk profiles?
**Method:** ANOVA, group-wise analysis
**Finding:** Occupation alone is not a strong predictor (negligible Cramer's V), but some occupations show slightly elevated risk due to income and payment pattern differences.

## 5. How does credit history age relate to fraud risk?
**Method:** Correlation analysis, SHAP dependence plots
**Finding:** Newer credit accounts (< 24 months) are disproportionately represented in the high-risk segment. Established credit histories (60+ months) are protective.

## 6. Can we build a model that catches 80%+ of risky profiles?
**Method:** LightGBM with class weights
**Finding:** Yes. Best model achieves ROC-AUC of 0.92 and recall of 0.83 at default threshold, catching 83% of high-risk profiles while maintaining reasonable precision.

## 7. What is the trade-off between false positives and false negatives?
**Method:** Threshold analysis, precision-recall curves
**Finding:** At threshold 0.5, precision is ~0.72 and recall is ~0.83. Lowering threshold to 0.35 increases recall to 0.92 but drops precision to 0.58. Optimal F1 threshold is around 0.45.

## 8. Are the model's decisions interpretable and trustworthy?
**Method:** SHAP waterfall plots, PDP plots
**Finding:** Yes. Individual predictions can be explained by specific feature contributions. For example, a high-risk prediction is typically driven by high utilization combined with frequent delayed payments, not by a single anomalous feature.
