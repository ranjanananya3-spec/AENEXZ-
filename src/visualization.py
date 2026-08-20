"""
Visualization Module.
Provides high-quality, production-ready plots using Seaborn and Matplotlib.
Includes correlation heatmaps, feature importance, actual vs. predicted, and residual diagnostics.
"""

import os
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Define global style constants for professional aesthetics
PRIMARY_COLOR = "#008080"    # Teal
SECONDARY_COLOR = "#FF6B6B"  # Soft Coral
ACCENT_COLOR = "#4D96FF"     # Sky Blue
NEUTRAL_DARK = "#2B2D42"     # Dark Slate
NEUTRAL_LIGHT = "#F8F9FA"    # Soft Off-White
GRID_COLOR = "#E2E8F0"       # Light Slate Grid

def set_premium_style() -> None:
    """
    Sets global styles for matplotlib to ensure a premium, modern design aesthetic.
    """
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
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#E2E8F0',
        'figure.autolayout': True
    })

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    target_col: str, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a Pearson correlation heatmap highlighting relationship with the target.
    """
    set_premium_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Custom diverging color palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
        mask=mask, 
        cmap=cmap, 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"}, 
        annot=True, 
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 9, "weight": "semibold"}
    )
    
    ax.set_title(f"Correlation Matrix (Target: {target_col})", pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to: {save_path}")
        
    return fig

def plot_vif_scores(
    vif_df: pd.DataFrame, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a horizontal bar chart of VIF scores with a collinearity threshold line.
    """
    set_premium_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter out intercept if present in DataFrame (e.g. 'const')
    vif_plot_df = vif_df[vif_df['Feature'] != 'const'].copy()
    
    # Order descending
    vif_plot_df = vif_plot_df.sort_values(by="VIF", ascending=True)
    
    # Define colors based on severity
    colors = [SECONDARY_COLOR if val >= 5 else PRIMARY_COLOR for val in vif_plot_df['VIF']]
    
    bars = ax.barh(vif_plot_df['Feature'], vif_plot_df['VIF'], color=colors, height=0.6)
    
    # Add VIF values on top of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.1, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}', 
            ha='left', 
            va='center', 
            fontsize=10, 
            fontweight='bold',
            color=NEUTRAL_DARK
        )
        
    # Draw threshold line
    ax.axvline(x=5, color='#E53E3E', linestyle='--', linewidth=1.5, label='Multicollinerity Threshold (VIF = 5)')
    ax.axvline(x=10, color='#9B2C2C', linestyle=':', linewidth=1.5, label='Severe Multicollinerity (VIF = 10)')
    
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("Variance Inflation Factor (VIF) by Feature", pad=15)
    ax.legend(loc="lower right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved VIF plot to: {save_path}")
        
    return fig

def plot_model_coefficients(
    coefficients: pd.Series, 
    model_name: str, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a horizontal bar chart ranking standardized model coefficients.
    """
    set_premium_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    coef_df = pd.DataFrame({
        'Feature': coefficients.index,
        'Coefficient': coefficients.values
    }).sort_values(by='Coefficient', key=abs, ascending=True)
    
    # Use green/teal for positive coefficients and red/coral for negative ones
    colors = [PRIMARY_COLOR if val >= 0 else SECONDARY_COLOR for val in coef_df['Coefficient']]
    
    bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, height=0.6)
    
    # Label each bar with value
    for bar in bars:
        width = bar.get_width()
        align = 'left' if width < 0 else 'right'
        offset = -0.05 if width < 0 else 0.05
        ax.text(
            width + offset, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}', 
            ha='left' if width >= 0 else 'right', 
            va='center', 
            fontsize=9, 
            fontweight='bold',
            color='white',
            bbox=dict(facecolor=PRIMARY_COLOR if width >= 0 else SECONDARY_COLOR, edgecolor='none', boxstyle='round,pad=0.2')
        )
        
    # Draw a vertical line at 0
    ax.axvline(x=0, color='#4A5568', linestyle='-', linewidth=0.8)
    
    ax.set_xlabel("Coefficient Weight (Standardized Scale)")
    ax.set_title(f"{model_name} - Standardized Coefficients Ranking", pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model coefficients to: {save_path}")
        
    return fig

def plot_actual_vs_predicted(
    y_true: pd.Series, 
    y_pred: np.ndarray, 
    model_name: str, 
    metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots actual values vs. predicted values with an ideal y=x line.
    """
    set_premium_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, color=PRIMARY_COLOR, alpha=0.6, edgecolors='white', s=50, label='Observations')
    
    # Ideal y=x line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=SECONDARY_COLOR, linestyle='--', linewidth=2, label='Ideal Fit (y = x)')
    
    # Trendline of actual vs predicted
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(np.unique(y_true), p(np.unique(y_true)), color=ACCENT_COLOR, linestyle='-', linewidth=1.5, label='Actual Trend')
    
    # Add metric annotations
    textstr = '\n'.join((
        f'$R^2$: {metrics["R2"]:.3f}',
        f'MAE: {metrics["MAE"]:.2f}',
        f'RMSE: {metrics["RMSE"]:.2f}'
    ))
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.9)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel("Actual Disease Progression Score")
    ax.set_ylabel("Predicted Disease Progression Score")
    ax.set_title(f"{model_name} - Actual vs. Predicted", pad=15)
    ax.legend(loc="lower right")
    ax.set_xlim([min_val - 10, max_val + 10])
    ax.set_ylim([min_val - 10, max_val + 10])
    ax.set_box_aspect(1)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved actual vs predicted plot to: {save_path}")
        
    return fig

def plot_residual_diagnostics(
    fitted: np.ndarray, 
    residuals: np.ndarray, 
    model_name: str, 
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots two panels: Residuals vs Fitted and a normal Q-Q plot of residuals.
    """
    set_premium_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Residuals vs Fitted
    ax1.scatter(fitted, residuals, color=PRIMARY_COLOR, alpha=0.6, edgecolors='white', s=50)
    ax1.axhline(y=0, color=SECONDARY_COLOR, linestyle='--', linewidth=1.5)
    
    # Lowess/smoothed line for homoscedasticity checking (local average line)
    # Using np.polyfit to draw a quadratic curve to look for non-linear patterns
    z = np.polyfit(fitted, residuals, 2)
    p = np.poly1d(z)
    fit_range = np.linspace(fitted.min(), fitted.max(), 100)
    ax1.plot(fit_range, p(fit_range), color=ACCENT_COLOR, linestyle='-', linewidth=1.5, label='Residual Trend')
    
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs. Fitted", pad=10)
    ax1.legend(loc="upper right")
    
    # Panel 2: Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color(PRIMARY_COLOR)  # Scatter points
    ax2.get_lines()[0].set_alpha(0.6)
    ax2.get_lines()[0].set_markersize(6)
    ax2.get_lines()[1].set_color(SECONDARY_COLOR)  # Line
    ax2.get_lines()[1].set_linewidth(1.5)
    
    ax2.set_title("Normal Q-Q Plot", pad=10)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    
    fig.suptitle(f"{model_name} - Residual Diagnostics", fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved residual diagnostics to: {save_path}")
        
    return fig
