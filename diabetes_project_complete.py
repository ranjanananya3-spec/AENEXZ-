"""
Diabetes Disease Progression Prediction - Complete Integrated Pipeline.

This single, self-contained Python script performs:
1. Data Ingestion & Validation (loads pre-scaled sklearn diabetes dataset).
2. Exploratory Data Analysis (Pearson correlation heatmap and VIF scoring).
3. Modeling (80/20 train/test split, Baseline Linear Regression, RidgeCV, and LassoCV).
4. Evaluation (R2, MAE, RMSE metrics).
5. Interpretability (Standardized coefficient bar charts).
6. Diagnostics (Actual vs. Predicted fit plots and Residual diagnostics vs. Fitted/Q-Q).

Usage:
    python3 diabetes_project_complete.py
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# =====================================================================
# 1. Visualization Styling Configuration
# =====================================================================
PRIMARY_COLOR = "#008080"    # Teal
SECONDARY_COLOR = "#FF6B6B"  # Soft Coral
ACCENT_COLOR = "#4D96FF"     # Sky Blue
NEUTRAL_DARK = "#2B2D42"     # Dark Slate
NEUTRAL_LIGHT = "#F8F9FA"    # Soft Off-White
GRID_COLOR = "#E2E8F0"       # Light Slate Grid

def set_premium_style() -> None:
    """Sets global matplotlib styles for premium, presentation-ready visualizations."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#CBD5E1',
        'axes.grid': True,
        'grid.color': GRID_COLOR,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.labelcolor': NEUTRAL_DARK,
        'xtick.color': NEUTRAL_DARK,
        'ytick.color': NEUTRAL_DARK,
        'text.color': NEUTRAL_DARK,
        'axes.titleweight': 'bold',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#E2E8F0',
        'figure.autolayout': True
    })

# =====================================================================
# 2. Data Ingestion & Integrity Validation
# =====================================================================
def load_and_validate_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and runs verification checks on the diabetes progression dataset.
    """
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    
    print("--- DATASET VERIFICATION ---")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nFeature Dtypes and Null Checks:")
    
    info_df = pd.DataFrame({
        'Dtype': X.dtypes,
        'Non-Null Count': X.notnull().sum(),
        'Null Count': X.isnull().sum()
    })
    print(info_df.to_string())
    
    assert X.isnull().sum().sum() == 0, "Error: Missing values found in features."
    assert y.isnull().sum() == 0, "Error: Missing values found in target."
    
    print("\nTarget Summary Statistics:")
    print(y.describe().to_string())
    print("\n*Note: Dataset features are pre-scaled and mean-centered by scikit-learn.")
    print("----------------------------\n")
    
    return X, y

# =====================================================================
# 3. Exploratory Data Analysis & Multicollinearity (VIF)
# =====================================================================
def run_multicollinearity_analysis(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) scores for features.
    """
    # VIF calculations require adding a constant intercept column
    X_const = add_constant(X, has_constant='add')
    vif_data = []
    
    # Calculate for all features excluding the const intercept
    features = [col for col in X_const.columns if col != 'const']
    for col in features:
        idx = X_const.columns.get_loc(col)
        vif = variance_inflation_factor(X_const.values, idx)
        vif_data.append({"Feature": col, "VIF": vif})
        
    return pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)

# =====================================================================
# 4. Model Training & Pipeline Cross-Validation
# =====================================================================
def train_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Trains baseline OLS, RidgeCV, and LassoCV models with cross-validation.
    """
    alphas = np.logspace(-4, 4, 200)
    
    # 1. Baseline Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # 2. Ridge Regression with Cross-Validation
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train, y_train)
    print(f"Optimal Ridge alpha: {ridge.alpha_:.5f}")
    
    # 3. Lasso Regression with Cross-Validation
    lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
    lasso.fit(X_train, y_train)
    print(f"Optimal Lasso alpha: {lasso.alpha_:.5f}")
    
    return {
        "Linear Regression": lr,
        "RidgeCV": ridge,
        "LassoCV": lasso
    }

# =====================================================================
# 5. Evaluation Metrics
# =====================================================================
def evaluate_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Computes R2, MAE, and RMSE for all models on both Train and Test splits.
    """
    results = []
    for name, model in models.items():
        for split, (features, targets) in [("Train", (X_train, y_train)), ("Test", (X_test, y_test))]:
            preds = model.predict(features)
            r2 = r2_score(targets, preds)
            mae = mean_absolute_error(targets, preds)
            rmse = np.sqrt(mean_squared_error(targets, preds))
            results.append({
                "Model": name,
                "Dataset": split,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse
            })
    return pd.DataFrame(results)

# =====================================================================
# 6. Reusable Visualization Plotters
# =====================================================================
def plot_correlation_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """Plots a Pearson correlation heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Pearson Correlation"},
                annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8, "weight": "semibold"})
    ax.set_title("Feature & Target Pearson Correlation Matrix", pad=15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_vif_chart(vif_df: pd.DataFrame, output_path: str) -> None:
    """Plots VIF horizontal bars with collinearity guidelines."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_df = vif_df.sort_values(by="VIF", ascending=True)
    colors = [SECONDARY_COLOR if val >= 5 else PRIMARY_COLOR for val in plot_df['VIF']]
    bars = ax.barh(plot_df['Feature'], plot_df['VIF'], color=colors, height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                ha='left', va='center', fontsize=9, fontweight='bold', color=NEUTRAL_DARK)
        
    ax.axvline(x=5, color='#E53E3E', linestyle='--', linewidth=1.2, label='Multicollinearity Threshold (VIF=5)')
    ax.axvline(x=10, color='#9B2C2C', linestyle=':', linewidth=1.2, label='Severe Multicollinearity (VIF=10)')
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("Variance Inflation Factors by Feature", pad=15)
    ax.legend(loc="lower right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_coefficients_ranking(coefs: pd.Series, model_name: str, output_path: str) -> None:
    """Plots horizontal bar charts showing relative feature importance weights."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    coef_df = pd.DataFrame({'Feature': coefs.index, 'Weight': coefs.values}).sort_values(by='Weight', key=abs, ascending=True)
    colors = [PRIMARY_COLOR if val >= 0 else SECONDARY_COLOR for val in coef_df['Weight']]
    bars = ax.barh(coef_df['Feature'], coef_df['Weight'], color=colors, height=0.6)
    
    for bar in bars:
        w = bar.get_width()
        align = 'left' if w < 0 else 'right'
        offset = -5 if w < 0 else 5
        ax.text(w + offset, bar.get_y() + bar.get_height()/2, f'{w:.1f}',
                ha='left' if w >= 0 else 'right', va='center', fontsize=8, fontweight='bold',
                color='white', bbox=dict(facecolor=PRIMARY_COLOR if w >= 0 else SECONDARY_COLOR, edgecolor='none', boxstyle='round,pad=0.2'))
        
    ax.axvline(x=0, color='#4A5568', linestyle='-', linewidth=0.8)
    ax.set_xlabel("Standardized Coefficient weight")
    ax.set_title(f"{model_name} Standardized Coefficients", pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fit_diagnostics(y_true: pd.Series, y_pred: np.ndarray, model_name: str, metrics: Dict[str, float], output_path: str) -> None:
    """Plots actual values vs predictions with reference lines and annotations."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, color=PRIMARY_COLOR, alpha=0.6, edgecolors='white', s=45, label='Observations')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=SECONDARY_COLOR, linestyle='--', linewidth=1.8, label='Ideal Fit (y=x)')
    
    # Trendline
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(np.unique(y_true), p(np.unique(y_true)), color=ACCENT_COLOR, linestyle='-', linewidth=1.2, label='Actual Fit')
    
    textstr = f'$R^2$: {metrics["R2"]:.3f}\nMAE: {metrics["MAE"]:.1f}\nRMSE: {metrics["RMSE"]:.1f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CBD5E1', alpha=0.9))
    
    ax.set_xlabel("Actual Progression Score")
    ax.set_ylabel("Predicted Progression Score")
    ax.set_title(f"{model_name}: Predictions vs. Actuals", pad=15)
    ax.legend(loc="lower right")
    ax.set_xlim([min_val - 10, max_val + 10])
    ax.set_ylim([min_val - 10, max_val + 10])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals_diagnostics(fitted: np.ndarray, residuals: np.ndarray, model_name: str, output_path: str) -> None:
    """Plots residuals vs fitted values and normal Q-Q plots for assumption checks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Panel 1: Residuals vs Fitted
    ax1.scatter(fitted, residuals, color=PRIMARY_COLOR, alpha=0.6, edgecolors='white', s=45)
    ax1.axhline(y=0, color=SECONDARY_COLOR, linestyle='--', linewidth=1.5)
    
    # Non-linear trend estimation line
    z = np.polyfit(fitted, residuals, 2)
    p = np.poly1d(z)
    fit_range = np.linspace(fitted.min(), fitted.max(), 100)
    ax1.plot(fit_range, p(fit_range), color=ACCENT_COLOR, linestyle='-', linewidth=1.5, label='Residual Trend')
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs. Fitted Values", pad=10)
    ax1.legend(loc="upper right")
    
    # Panel 2: Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color(PRIMARY_COLOR)
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[0].set_markersize(5)
    ax2.get_lines()[1].set_color(SECONDARY_COLOR)
    ax2.get_lines()[1].set_linewidth(1.5)
    ax2.set_title("Normal Q-Q Plot of Residuals", pad=10)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    
    fig.suptitle(f"{model_name}: Residual Diagnostic Checking", fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 7. Main Pipeline Orchestrator
# =====================================================================
def main() -> None:
    # Setup styling and directories
    set_premium_style()
    output_dir = "outputs_complete"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("RUNNING THE INTEGRATED DIABETES PROGRESSION MACHINE LEARNING PIPELINE")
    print("=" * 70)
    
    # 1. Ingestion & Preprocessing Verification
    X, y = load_and_validate_data()
    df_full = X.copy()
    df_full['target'] = y
    
    # 2. EDA & Feature Correlation Heatmap
    print("[Step 1/5] Running correlation analysis and saving matrix...")
    heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(df_full, heatmap_file)
    
    # 3. Multicollinearity VIF Analysis
    print("[Step 2/5] Assessing multicollinearity using VIF...")
    vif_results = run_multicollinearity_analysis(X)
    print("Computed Variance Inflation Factors (VIF):")
    print(vif_results.to_string(index=False))
    
    vif_file = os.path.join(output_dir, "vif_scores.png")
    plot_vif_chart(vif_results, vif_file)
    
    # 4. Data Splitting & Cross-Validated Model Training
    print("\n[Step 3/5] Splitting data into 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting baseline Ordinary Least Squares (OLS) Regression...")
    print("Fitting RidgeCV & LassoCV models with 5-fold Cross-Validation...")
    models = train_pipeline(X_train, y_train)
    
    # 5. Model Evaluation and Diagnostic Plots
    print("\n[Step 4/5] Evaluating performance metrics...")
    comparison_table = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    print("\n======================================================================")
    print("FINAL PERFORMANCE COMPARISON")
    print("======================================================================")
    print(comparison_table.to_string(index=False))
    print("======================================================================")
    
    print("\n[Step 5/5] Generating diagnostic plots for each model...")
    for name, model in models.items():
        # Extrapolate metrics for annotations
        test_metrics_row = comparison_table[(comparison_table['Model'] == name) & (comparison_table['Dataset'] == 'Test')].iloc[0]
        test_metrics = {"R2": test_metrics_row["R2"], "MAE": test_metrics_row["MAE"], "RMSE": test_metrics_row["RMSE"]}
        
        # Save coefficients ranking chart
        if hasattr(model, "coef_"):
            coef_series = pd.Series(model.coef_, index=X.columns)
            coef_file = os.path.join(output_dir, f"coefficients_{name.lower().replace(' ', '_')}.png")
            plot_coefficients_ranking(coef_series, name, coef_file)
            
        # Save predictions vs actuals scatter fit plot
        fit_file = os.path.join(output_dir, f"predictions_fit_{name.lower().replace(' ', '_')}.png")
        plot_fit_diagnostics(y_test, model.predict(X_test), name, test_metrics, fit_file)
        
        # Save residuals plots (Homoscedasticity check and Normal Q-Q check)
        fitted = model.predict(X_test)
        residuals = y_test.values - fitted
        res_file = os.path.join(output_dir, f"residuals_diagnostics_{name.lower().replace(' ', '_')}.png")
        plot_residuals_diagnostics(fitted, residuals, name, res_file)
        
    print(f"\nPipeline successfully run! All outputs are saved to the '{output_dir}/' folder.")
    print("=" * 70)

if __name__ == "__main__":
    main()
