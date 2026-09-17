paisabazaar-fraud-detection-xai/
|-- data/
|   `-- dataset.csv                    # PaisaBazaar credit scoring dataset (100K records, 30MB)
|-- notebooks/
|   `-- fraud_detection_analysis.ipynb  # Main notebook (fully executed with outputs)
|-- src/
|   |-- __init__.py                    # Package init
|   |-- config.py                      # Constants, paths, hyperparameters
|   |-- utils.py                       # Logging, seeding, helpers
|   |-- data_loader.py                 # CSV loading and audit
|   |-- data_cleaning.py               # Cleaning pipeline
|   |-- feature_engineering.py         # Feature engineering pipeline
|   |-- eda.py                         # Visualization functions
|   |-- statistical_analysis.py        # Hypothesis tests, correlations
|   |-- modeling.py                    # ML training, evaluation, plots
|   `-- explainability.py             # SHAP, PDP, permutation importance
|-- outputs/
|   |-- figures/                       # All generated plots (37 PNG files)
|   |-- interactive/                   # 11 interactive Plotly HTML dashboards
|   `-- models/                        # Saved model files (6 joblib, gitignored)
|-- README.md                          # Project overview and results
|-- requirements.txt                   # Python dependencies
|-- environment.yml                    # Conda environment
|-- Dockerfile                         # Docker support
|-- LICENSE                            # MIT License
|-- CONTRIBUTING.md                    # Contribution guidelines
|-- CHANGELOG.md                       # Version history
|-- project_structure.md               # This file
|-- dataset_description.md             # Dataset details
|-- business_questions.md              # Business questions answered
`-- model_card.md                      # Model documentation
```
