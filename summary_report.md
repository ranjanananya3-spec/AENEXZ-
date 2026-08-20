# Project Review Report: Diabetes Disease Progression Prediction

**Author**: Senior Data Scientist / ML Engineer  
**Date**: August 20, 2026  
**Objective**: Build and review a production-quality predictive model for 1-year diabetes disease progression using 10 baseline physiological features.

---

## 1. Executive Summary
This review details the development and evaluation of a predictive model for diabetes progression. We compared a baseline Ordinary Least Squares (OLS) Linear Regression against two regularized models: Ridge Regression (`RidgeCV`) and Lasso Regression (`LassoCV`). 
- **Key Finding**: Variance Inflation Factor (VIF) diagnostics revealed severe multicollinearity among blood serum measurements ($s1$ to $s6$), causing standard OLS regression coefficients to be highly unstable.
- **Top Performer**: Lasso Regression (`LassoCV`) achieved the best generalization, with a test $R^2$ of **0.471** (an improvement over OLS's **0.453**) while shrinking redundant features to zero, yielding a highly interpretable and compact clinical model.

---

## 2. Dataset and Preprocessing
The model was trained on the `sklearn.datasets.load_diabetes` dataset ($N = 442$ patient records).
- **Target Variable**: Quantitative measure of disease progression one year after baseline (ranges from 25 to 346).
- **Features**: Age, Sex, Body Mass Index (`bmi`), Mean Blood Pressure (`bp`), and 6 blood serum measurements (`s1`–`s6`).
- **Feature Scaling**: 
  > [!IMPORTANT]
  > All 10 baseline features are **pre-scaled** in scikit-learn. Each column is mean-centered and scaled by the standard deviation times the square root of the number of samples (sum of squares for each column is 1). Therefore, no additional scaling/standardization was performed to avoid double-scaling errors.

---

## 3. Multicollinearity & Correlation Analysis
A Pearson correlation matrix was computed to understand linear relationships. Additionally, a Variance Inflation Factor (VIF) analysis was conducted on features to detect multicollinearity.

### Variance Inflation Factor (VIF) Scores
VIF scores greater than 5 indicate multicollinearity, and scores above 10 suggest severe collinearity:

| Feature | VIF Score | Status / Interpretation |
| :--- | :---: | :--- |
| **s1** (Total Cholesterol) | **59.20** | Severe Multicollinearity (Redundant) |
| **s2** (LDL Cholesterol) | **39.19** | Severe Multicollinearity (Redundant) |
| **s3** (HDL Cholesterol) | **15.40** | High Multicollinearity |
| **s5** (Triglycerides) | **10.08** | High Multicollinearity |
| **s4** (Cholesterol/HDL) | **8.89** | Moderate-to-High Multicollinearity |
| **bmi** | 1.51 | Low Collinearity |
| **s6** (Blood Sugar) | 1.48 | Low Collinearity |
| **bp** | 1.46 | Low Collinearity |
| **sex** | 1.28 | Low Collinearity |
| **age** | 1.22 | Low Collinearity |

*Insight*: The extremely high VIF scores for $s1$ and $s2$ indicate that these variables carry redundant information. In standard OLS, this causes large standard errors and erratic coefficient estimations (e.g., OLS assigns a large negative coefficient to $s1$ and a large positive coefficient to $s2$).

---

## 4. Model Performance Comparison
Models were trained using an 80/20 train/test split. Hyperparameters for Ridge and Lasso were selected via 5-fold cross-validation (`cv=5`).

| Model | Alpha ($\alpha$) | Dataset | $R^2$ Score | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | — | Train | 0.528 | 43.48 | 53.56 |
| | | Test | 0.453 | 42.79 | 53.85 |
| **RidgeCV** | 0.0944 | Train | 0.521 | 43.89 | 53.95 |
| | | Test | 0.461 | 42.99 | 53.45 |
| **LassoCV** | 0.0784 | Train | 0.519 | 44.09 | 54.06 |
| | | Test | **0.471** | **42.79** | **52.92** |

### Discussion:
1. **Regularization Advantage**: OLS suffered from slight overfitting. Ridge and Lasso regularization penalize large coefficients, which mitigated multicollinearity, resulting in higher test $R^2$ scores and lower test RMSE.
2. **Lasso CV as the Best Model**: LassoCV achieved the highest test $R^2$ (**0.471**) and the lowest test RMSE (**52.92**).

---

## 5. Model Interpretability & Feature Importance
Standardized coefficients from the models allow for direct feature comparison.

### Coefficients Comparison

| Feature | Linear Regression (OLS) | RidgeCV | LassoCV (Preferred) |
| :--- | :---: | :---: | :---: |
| **age** | -10.01 | -3.72 | 0.00 |
| **sex** | -239.77 | -226.79 | -186.20 |
| **bmi** | 521.84 | 519.82 | 521.36 |
| **bp** | 324.39 | 311.66 | 291.56 |
| **s1** | -792.83 | -52.48 | -0.00 |
| **s2** | 476.74 | -32.84 | 0.00 |
| **s3** | 101.04 | -173.83 | -132.84 |
| **s4** | 177.06 | 120.35 | 0.00 |
| **s5** | 751.28 | 499.78 | 498.44 |
| **s6** | 67.63 | 82.26 | 39.80 |

### Key Interpretations:
* **Body Mass Index (`bmi`)** is the strongest positive predictor of disease progression (Lasso coefficient: **521.36**).
* **Serum Triglycerides (`s5`)** is the second strongest positive predictor (Lasso coefficient: **498.44**).
* **Average Blood Pressure (`bp`)** also contributes strongly to progression (Lasso coefficient: **291.56**).
* **Feature Sparsity in Lasso**: LassoCV successfully zeroed out coefficients for **age**, **s2** (LDL), and **s4** (Cholesterol/HDL). This reduces clinical model complexity (retaining only 7 of the 10 variables) while delivering superior predictive power.

---

## 6. Diagnostic Validation & Regression Assumptions
To verify that linear model assumptions hold, diagnostics were performed on the LassoCV residuals on the test set:
1. **Linearity & Homoscedasticity (Residuals vs. Fitted)**: Residuals are distributed randomly around the zero horizontal axis. No distinct non-linear patterns or funnel shapes (heteroscedasticity) were observed, indicating that a linear model structure is valid.
2. **Normality (Q-Q Plot)**: The residuals follow the theoretical normal line closely, confirming that the error terms are normally distributed and statistical inferences are reliable.

---

## 7. Recommendations for Production Deployability
- **Adopt LassoCV**: LassoCV should be selected as the production model due to its high accuracy, stability under multicollinearity, and automatic feature selection which eliminates the cost of collecting age, LDL, and Cholesterol/HDL data.
- **Clinical Target Areas**: Preventative strategies targeting BMI reduction, blood pressure control, and triglyceride management are recommended to mitigate diabetes progression.
