"""
Modeling Module.
Handles the train-test splitting and trains Linear Regression, RidgeCV, and LassoCV.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Fits a baseline Ordinary Least Squares (OLS) Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_cv(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    alphas: np.ndarray = np.logspace(-4, 4, 200),
    cv: int = 5
) -> RidgeCV:
    """
    Fits a Ridge regression model with cross-validated alpha selection.
    """
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    return model

def train_lasso_cv(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    alphas: np.ndarray = np.logspace(-4, 4, 200),
    cv: int = 5,
    max_iter: int = 10000
) -> LassoCV:
    """
    Fits a Lasso regression model with cross-validated alpha selection.
    """
    model = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_all_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Orchestrates the training of Linear Regression, RidgeCV, and LassoCV.
    
    Returns:
        Dict containing model name as key and fitted model object as value.
    """
    print("Training Baseline Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    
    print("Training RidgeCV Regression...")
    ridge_model = train_ridge_cv(X_train, y_train)
    print(f"Optimal Ridge alpha: {ridge_model.alpha_:.5f}")
    
    print("Training LassoCV Regression...")
    lasso_model = train_lasso_cv(X_train, y_train)
    print(f"Optimal Lasso alpha: {lasso_model.alpha_:.5f}")
    
    return {
        "Linear Regression": lr_model,
        "RidgeCV": ridge_model,
        "LassoCV": lasso_model
    }
