"""
Main Pipeline Orchestrator.
Runs the entire machine learning pipeline end-to-end, printing statistical outputs
and saving high-quality diagnostic visualizations.
"""

import os
import sys
import pandas as pd

# Add the project root to the python path so src modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import load_diabetes_data
from src.eda import compute_pearson_correlation, calculate_vif
from src.modeling import split_data, train_all_models
from src.evaluation import evaluate_model, generate_comparison_table, get_residuals_diagnostics
from src.visualization import (
    plot_correlation_heatmap,
    plot_vif_scores,
    plot_model_coefficients,
    plot_actual_vs_predicted,
    plot_residual_diagnostics
)

def run_pipeline(output_dir: str = "outputs") -> None:
    """
    Orchestrates the entire ML pipeline from data ingestion to model diagnostics and visualization.
    """
    print("=" * 60)
    print("STARTING DIABETES PROGRESSION PREDICTION PIPELINE")
    print("=" * 60)
    
    # 1. Ingestion
    print("\n[Step 1] Loading and validating dataset...")
    X, y = load_diabetes_data()
    
    # Merge into single dataframe for EDA
    df_full = X.copy()
    df_full['target'] = y
    
    # 2. EDA & Feature Correlation
    print("\n[Step 2] Conducting EDA and correlation analysis...")
    corr_matrix = compute_pearson_correlation(df_full)
    
    # Save correlation heatmap
    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plot_correlation_heatmap(df_full, target_col="target", save_path=heatmap_path)
    
    # Compute and save VIF
    print("\nCalculating Variance Inflation Factors (VIF) to inspect multicollinearity...")
    vif_df = calculate_vif(X, include_constant=True)
    print("Variance Inflation Factor (VIF) Scores:")
    print(vif_df.to_string(index=False))
    
    vif_plot_path = os.path.join(output_dir, "vif_scores.png")
    plot_vif_scores(vif_df, save_path=vif_plot_path)
    
    # 3. Model Training
    print("\n[Step 3] Splitting data and training regression models...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    models = train_all_models(X_train, y_train)
    
    # 4. Evaluation and Visualizations
    print("\n[Step 4] Evaluating models and generating diagnostic plots...")
    eval_results = {}
    
    for model_name, model in models.items():
        # Evaluate model
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        eval_results[model_name] = metrics
        
        # Plot coefficients
        if hasattr(model, "coef_"):
            coef_series = pd.Series(model.coef_, index=X.columns)
            coef_path = os.path.join(output_dir, f"coef_{model_name.replace(' ', '_').lower()}.png")
            plot_model_coefficients(coef_series, model_name, save_path=coef_path)
            
        # Plot actual vs predicted
        act_vs_pred_path = os.path.join(output_dir, f"act_vs_pred_{model_name.replace(' ', '_').lower()}.png")
        plot_actual_vs_predicted(
            y_test, 
            model.predict(X_test), 
            model_name, 
            metrics["Test"], 
            save_path=act_vs_pred_path
        )
        
        # Plot residuals
        fitted_val, res = get_residuals_diagnostics(model, X_test, y_test)
        residuals_path = os.path.join(output_dir, f"residuals_{model_name.replace(' ', '_').lower()}.png")
        plot_residual_diagnostics(fitted_val, res, model_name, save_path=residuals_path)
        
    # 5. Model Comparison
    print("\n[Step 5] Consolidating results...")
    comparison_table = generate_comparison_table(eval_results)
    
    # Filter for cleaner comparison views
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_table.to_string(index=False))
    print("=" * 60)
    
    # Write a summary report text file
    write_summary_report(comparison_table, vif_df, output_dir)
    
    print("\nPipeline execution complete. All outputs are saved to the 'outputs/' folder.")
    print("=" * 60)

def write_summary_report(comp_table: pd.DataFrame, vif_df: pd.DataFrame, output_dir: str) -> None:
    """
    Writes a summary text file with key findings.
    """
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DIABETES PROGRESSION PREDICTION PIPELINE SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. DATA CHARACTERISTICS\n")
        f.write("- Target variable: Quantitative measure of disease progression one year after baseline.\n")
        f.write("- Features: Age, Sex, Body Mass Index (BMI), Average Blood Pressure (BP),\n")
        f.write("  and 6 blood serum measurements (S1, S2, S3, S4, S5, S6).\n")
        f.write("- Note: Features are already mean-centered and scaled in the scikit-learn dataset.\n\n")
        
        f.write("2. MULTICOLLINEARITY ANALYSIS (VIF Scores)\n")
        f.write("- A VIF > 5 indicates potential multicollinearity, while VIF > 10 is severe.\n")
        f.write("- In this dataset, several blood serum measurements exhibit severe multicollinearity:\n")
        for _, row in vif_df.iterrows():
            if row['Feature'] != 'const':
                f.write(f"  * {row['Feature']}: {row['VIF']:.2f}\n")
        f.write("\nKey Observations on Collinearity:\n")
        f.write("- s1 and s2 exhibit extremely high VIF scores (> 50 and > 30), indicating redundancy.\n")
        f.write("- This multicollinearity inflates standard error of coefficients in Ordinary Linear Regression,\n")
        f.write("  making coefficient estimation unstable. Regularization (L1/L2) is strongly recommended.\n\n")
        
        f.write("3. PERFORMANCE METRICS COMPARISON\n")
        f.write(comp_table.to_string(index=False) + "\n\n")
        
        # Analyze performance difference
        test_r2_lr = comp_table[(comp_table['Model'] == 'Linear Regression') & (comp_table['Dataset'] == 'Test')]['R2'].values[0]
        test_r2_ridge = comp_table[(comp_table['Model'] == 'RidgeCV') & (comp_table['Dataset'] == 'Test')]['R2'].values[0]
        test_r2_lasso = comp_table[(comp_table['Model'] == 'LassoCV') & (comp_table['Dataset'] == 'Test')]['R2'].values[0]
        
        f.write("4. KEY FINDINGS & INSIGHTS\n")
        f.write(f"- Baseline Linear Regression test R²: {test_r2_lr:.4f}\n")
        f.write(f"- RidgeCV test R²: {test_r2_ridge:.4f}\n")
        f.write(f"- LassoCV test R²: {test_r2_lasso:.4f}\n")
        f.write("- Regularization (Ridge and Lasso) helps address multicollinearity and improves coefficient stability.\n")
        f.write("- LassoCV performs automatic feature selection by setting less important coefficients to zero,\n")
        f.write("  which yields a simpler and more interpretable model while maintaining comparable predictive power.\n")
        f.write("- Top predictors usually include BMI, s5 (Triglycerides), and BP.\n")
        f.write("=" * 80 + "\n")
        
    print(f"Summary report written to: {report_path}")

if __name__ == "__main__":
    run_pipeline()
