"""Quick test script to verify all modules work end-to-end."""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress interactive plotting when running as script
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

import plotly.io as pio
pio.renderers.default = "png"

from src.config import RANDOM_SEED
from src.utils import set_all_seeds
set_all_seeds(RANDOM_SEED)

print("=" * 60)
print("STAGE 1: Data Loading")
print("=" * 60)
from src.data_loader import load_data
df = load_data()
print(f"OK: {df.shape}")

print("\n" + "=" * 60)
print("STAGE 2: Data Cleaning")
print("=" * 60)
from src.data_cleaning import run_cleaning_pipeline
df_clean = run_cleaning_pipeline(df)
print(f"OK: {df_clean.shape}")

print("\n" + "=" * 60)
print("STAGE 3: Feature Engineering")
print("=" * 60)
from src.feature_engineering import run_feature_engineering
df_feat = run_feature_engineering(df_clean)
print(f"OK: {df_feat.shape}")

print("\n" + "=" * 60)
print("STAGE 4: ML Pipeline (Class Weights)")
print("=" * 60)
from src.modeling import run_full_ml_pipeline
ml_results = run_full_ml_pipeline(df_feat, target='Is_Fraud', imbalance_method='class_weights')
print(f"\nBest model: {ml_results['best_model_name']}")
print(f"Metrics:\n{ml_results['metrics'][['Model', 'ROC-AUC', 'F1', 'Recall', 'Precision']].to_string()}")

print("\n" + "=" * 60)
print("STAGE 5: XAI Pipeline")
print("=" * 60)
from src.explainability import run_full_xai_pipeline
best_model = ml_results['models'][ml_results['best_model_name']]
data = ml_results['data']
xai_results = run_full_xai_pipeline(
    best_model, data['X_train'], data['X_test'], data['y_test'],
    model_name=ml_results['best_model_name']
)
print("XAI OK")

print("\n" + "=" * 60)
print("ALL STAGES COMPLETE")
print("=" * 60)
