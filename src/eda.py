"""
Exploratory Data Analysis (EDA) Module.
Computes summary statistics, generates correlations, and calculates
Variance Inflation Factors (VIF) to detect and measure multicollinearity.
"""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics for a DataFrame.
    """
    return df.describe()

def compute_pearson_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the Pearson correlation matrix for the DataFrame.
    """
    return df.corr(method='pearson')

def calculate_vif(df: pd.DataFrame, include_constant: bool = True) -> pd.DataFrame:
    """
    Computes the Variance Inflation Factor (VIF) for the features in the DataFrame
    to identify multicollinearity.
    
    Args:
        df (pd.DataFrame): Dataframe of features.
        include_constant (bool): Whether to add a constant column to the data (recommended).
        
    Returns:
        pd.DataFrame: Dataframe containing features and their corresponding VIF scores.
    """
    feature_df = df.copy()
    if include_constant:
        # Add constant to represent intercept in VIF calculation
        feature_df = add_constant(feature_df, has_constant='add')
        
    vif_data = []
    # Calculate VIF for each column (excluding constant if added)
    cols_to_calculate = [col for col in feature_df.columns if col != 'const']
    
    for col in cols_to_calculate:
        col_index = feature_df.columns.get_loc(col)
        vif = variance_inflation_factor(feature_df.values, col_index)
        vif_data.append({"Feature": col, "VIF": vif})
        
    return pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)
