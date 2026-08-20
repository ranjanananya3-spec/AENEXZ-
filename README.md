# Diabetes Disease Progression Prediction Pipeline

This repository contains a modular, production-ready machine learning pipeline and an interactive Jupyter Notebook designed to predict a quantitative measure of diabetes disease progression after one year, based on 10 baseline physiological features.

The analysis specifically addresses multicollinearity among blood serum measurements and compares ordinary least squares (OLS) regression against cross-validated L1 (Lasso) and L2 (Ridge) regularization.

## Project Structure

```text
diabetes_prediction/
├── README.md                    # Project overview and instructions
├── requirements.txt             # Python packages required
├── main.py                      # Main entrypoint script to execute the pipeline
├── notebooks/
│   └── diabetes_analysis.ipynb  # Interactive walkthrough with pre-rendered outputs & plots
├── src/                         # Reusable library modules
│   ├── __init__.py
│   ├── data_ingestion.py        # Dataset loading & verification
│   ├── eda.py                   # Summaries, correlations, and VIF calculation
│   ├── modeling.py              # Train/test split and CV model training
│   ├── evaluation.py            # R², MAE, RMSE metrics calculation
│   └── visualization.py         # Premium visualizations (plots & charts)
└── outputs/                     # Generated diagnostic plots and text report
    ├── summary_report.txt       # Tabulated summary of results & insights
    ├── correlation_heatmap.png  # Pearson feature correlation heatmap
    ├── vif_scores.png           # Feature VIF score bar chart
    ├── coef_*.png               # Feature coefficients rankings for each model
    ├── act_vs_pred_*.png        # Predicted vs. actual plots with y=x baseline
    └── residuals_*.png          # Residuals vs fitted and Q-Q diagnostic plots
```

## Features

1. **Robust Data Validation**: Auto-verifies target characteristics, feature types, shapes, and checks for null values.
2. **Multicollinearity Diagnostic**: Uses statsmodels to calculate the **Variance Inflation Factor (VIF)** of all baseline variables to spot redundant features.
3. **Regularized Cross-Validation**: Employs `RidgeCV` and `LassoCV` with 5-fold cross-validation to search over alpha values and solve collinearity issues.
4. **Residual Diagnostics**: Validates linear regression assumptions (linearity, normality, and homoscedasticity) via Residual vs. Fitted plots and Normal Q-Q plots.
5. **Premium Visualization Styles**: Modern teal/coral styling applied across all heatmaps, charts, and plots.

## Installation

1. Set up a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the ML Pipeline Script

Execute the main orchestrator script to load the data, train all models, print results, and save diagnostic plots under the `outputs/` folder:

```bash
python3 main.py
```

### Run the Jupyter Notebook

Launch Jupyter Notebook or Jupyter Lab to inspect the pre-rendered analysis or run it cell-by-cell:

```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

## Modeling Results Summary

| Model | Dataset | $R^2$ Score | MAE | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Train | 0.528 | 43.48 | 53.56 |
| **Linear Regression** | Test | 0.453 | 42.79 | 53.85 |
| **RidgeCV** | Train | 0.521 | 43.89 | 53.95 |
| **RidgeCV** (Best Alpha: 0.094) | Test | **0.461** | 42.99 | 53.45 |
| **LassoCV** | Train | 0.519 | 44.09 | 54.06 |
| **LassoCV** (Best Alpha: 0.078) | Test | **0.471** | 42.79 | 52.92 |

### Key Findings
- **Multicollinearity**: Blood serum measurements `s1` (VIF $\approx 59.2$) and `s2` (VIF $\approx 39.2$) display massive multicollinearity.
- **Regularization**: Addressing this multicollinearity via `LassoCV` and `RidgeCV` improved generalization on the test set. `LassoCV` achieved the highest test $R^2$ of **0.471**.
- **Clinical Predictors**: Body Mass Index (`bmi`), blood pressure (`bp`), and serum triglycerides (`s5`) are the most dominant baseline indicators of progression.
- **Sparsity**: `LassoCV` successfully zeroed out coefficients for redundant or non-predictive features (`age`, `s2`, `s4`), offering a highly compact and interpretable clinical model.
