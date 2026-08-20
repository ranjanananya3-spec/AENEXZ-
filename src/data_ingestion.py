"""
Data Ingestion Module.
Responsible for loading the scikit-learn diabetes dataset, verifying types, shapes,
missing values, and standard dataset characteristics.
"""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_diabetes

def load_diabetes_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the diabetes dataset from sklearn.datasets.
    
    Returns:
        X (pd.DataFrame): The physiological features (scaled).
        y (pd.Series): The disease progression target variable after one year.
    """
    # Load dataset as a pandas DataFrame
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target
    
    # Run data verification checks
    verify_dataset(X, y)
    
    return X, y

def verify_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Validates dataset shape, types, missing values, and scale characteristics.
    
    Args:
        X (pd.DataFrame): Features dataframe.
        y (pd.Series): Target series.
    """
    print("--- DATASET VERIFICATION ---")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print("\nFeature Types and Missing Values:")
    
    # Check for missing values and types
    info_df = pd.DataFrame({
        'Dtype': X.dtypes,
        'Non-Null Count': X.notnull().sum(),
        'Null Count': X.isnull().sum()
    })
    print(info_df)
    
    # Assert there are no missing values
    assert X.isnull().sum().sum() == 0, "Warning: Missing values found in features."
    assert y.isnull().sum() == 0, "Warning: Missing values found in target."
    
    print("\nTarget Summary Statistics:")
    print(y.describe())
    
    print("\nNote: Scikit-learn's diabetes dataset features are pre-scaled.")
    print("Each of the 10 feature variables have been mean-centered and scaled by the ")
    print("standard deviation times the square root of the number of samples ")
    print("(i.e. the sum of squares of each column is 1).")
    print("----------------------------\n")
