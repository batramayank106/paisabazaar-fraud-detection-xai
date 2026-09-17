# Model Card

## Model Details
- **Model:** LightGBM (Light Gradient Boosting Machine)
- **Version:** 1.0
- **Date:** December 2025
- **Author:** Mayank Batra
- **Framework:** scikit-learn compatible API via LightGBM

## Intended Use
- **Primary:** Flagging high-risk credit profiles for manual review
- **Out-of-scope:** Not a standalone fraud decision system. Outputs should be used as risk scores to prioritize investigation, not as binary accept/reject decisions.

## Training Data
- 80,000 customer-month observations (80% of 100K dataset)
- Synthetic fraud labels engineered from domain rules on Poor credit segment
- Class imbalance handled via class weights (ratio 1:4)

## Evaluation Data
- 20,000 customer-month observations (20% holdout, stratified)

## Performance Metrics

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9221 |
| PR-AUC | 0.8104 |
| Accuracy | 0.8733 |
| Precision | 0.7187 |
| Recall | 0.8314 |
| F1-Score | 0.7710 |
| Cohen's Kappa | 0.6756 |
| Matthews Correlation Coefficient | 0.6782 |

## Fairness Considerations
- Model is trained on synthetic labels derived from credit score and behavioral features
- No protected attributes (gender, race, religion) are used as features
- Occupation is included but shows negligible predictive power (Cramer's V < 0.1)
- Model should be monitored for disparate impact across demographic groups in production

## Limitations
- Synthetic target may not perfectly reflect real-world fraud patterns
- Data is from a single lending platform; generalizability to other populations is untested
- Temporal drift not assessed (dataset spans 8 months only)
- Model does not account for external factors (economic conditions, regulatory changes)

## Hyperparameters
```
class_weight: balanced
learning_rate: 0.05
n_estimators: 500
max_depth: 7
num_leaves: 31
min_child_samples: 20
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 1.0
random_state: 42
```

## Ethical Considerations
- Credit scoring models can have significant impact on people's financial access
- False negatives (missed fraud) result in financial loss to the institution
- False positives (false alarms) may unnecessarily restrict credit access for legitimate customers
- Regular auditing for bias and drift is recommended in any production deployment
