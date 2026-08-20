"""
Evaluation Module.
Computes performance metrics (R², MAE, RMSE) on train and test sets
and prepares residual data for diagnostics.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes R2, MAE, and RMSE for predictions.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }

def evaluate_model(
    model: Any, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates a single model on both training and test data.
    
    Returns:
        Dict with "Train" and "Test" keys containing their respective performance metrics.
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    
    return {
        "Train": train_metrics,
        "Test": test_metrics
    }

def generate_comparison_table(
    evaluation_results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Consolidates evaluation metrics from multiple models into a clean summary DataFrame.
    
    Args:
        evaluation_results: Dict structure: {model_name: {"Train": metrics, "Test": metrics}}
        
    Returns:
        pd.DataFrame: Tabulated comparison of model performances.
    """
    rows = []
    for model_name, sets in evaluation_results.items():
        for set_name, metrics in sets.items():
            rows.append({
                "Model": model_name,
                "Dataset": set_name,
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"]
            })
            
    return pd.DataFrame(rows)

def get_residuals_diagnostics(
    model: Any, 
    X: pd.DataFrame, 
    y: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates predicted (fitted) values and residuals.
    
    Returns:
        fitted_values (np.ndarray): Predicted values.
        residuals (np.ndarray): Difference between actual and predicted values.
    """
    predictions = model.predict(X)
    residuals = y.values - predictions
    return predictions, residuals
