# Executive Abstract: Predictive Modeling of Diabetes Disease Progression

**Project Title**: Linear & Regularized Clinical Modeling of One-Year Diabetes Progression  
**Author**: Senior Data Scientist & ML Engineer  
**Dataset**: Scikit-Learn Diabetes Cohort ($N = 442$)  

---

## I. Clinical Context and Predictive Objective
Predicting the rate of diabetes progression is vital for personalizing patient care, optimizing clinical workflows, and allocating healthcare resources. This project builds and analyzes a production-quality predictive model designed to estimate a quantitative index of diabetes disease progression one year after baseline measurements. 

The baseline features include 10 physiological variables: age, sex, body mass index (BMI), mean blood pressure (BP), and six blood serum measurements ($s1$, $s2$, $s3$, $s4$, $s5$, $s6$). 

A primary focus of this research is balancing predictive accuracy with model interpretability, ensuring that model weights align with clinical expectations while maintaining robustness against statistical anomalies like multicollinearity.

---

## II. Preprocessing & Data Characteristics
The study utilizes the canonical diabetes patient cohort, comprising 442 records with no missing values. 

A critical characteristic of this dataset is that all features have been **pre-scaled**: they are mean-centered and scaled by the standard deviation times the square root of the number of samples. This ensures that the sum of squares of each feature column is equal to 1. Consequently, standardizing features during model training is unnecessary and would lead to incorrect double-scaling.

The target variable is a continuous quantitative measure of disease progression one year after baseline, exhibiting a mean of 152.13 and a standard deviation of 77.09.

---

## III. The Multicollinearity Challenge: VIF Analysis
An ordinary least squares (OLS) linear model struggles in clinical settings when features exhibit strong correlation. A Variance Inflation Factor (VIF) diagnostic was computed to inspect relationships within the blood serum metrics ($s1$ to $s6$).

The VIF scores revealed severe multicollinearity:
* **$s1$ (Total Cholesterol)**: VIF = **59.20**
* **$s2$ (LDL)**: VIF = **39.19**
* **$s3$ (HDL)**: VIF = **15.40**
* **$s5$ (Triglycerides)**: VIF = **10.08**
* **$s4$ (Cholesterol/HDL Ratio)**: VIF = **8.89**

Because the VIF values for $s1$, $s2$, $s3$, and $s5$ far exceed the conservative multicollinearity threshold of 5, OLS coefficient estimates are highly unstable, exhibiting inflated standard errors. This is visible in OLS, which assigns a massive negative weight (-792.83) to $s1$ and a massive positive weight (+476.74) to $s2$. To address this, we employed L1 (Lasso) and L2 (Ridge) regularization to stabilize and generalize the model.

---

## IV. Comparative Methodology & Hyperparameter Selection
The dataset was split using an 80/20 train/test partition to evaluate generalization on unseen data. Three modeling approaches were evaluated:
1. **Ordinary Least Squares (OLS) Regression**: Baseline model without penalty.
2. **Ridge Regression (L2 Penalty)**: Shrinks coefficient magnitudes, distributing weights among collinear features. Hyperparameter selection ($\alpha = 0.0944$) was optimized via 5-fold cross-validation.
3. **Lasso Regression (L1 Penalty)**: Shrinks coefficients to exactly zero, introducing sparsity and performing automatic feature selection. Hyperparameter selection ($\alpha = 0.0784$) was optimized via 5-fold cross-validation.

### Performance Results Table
The models were evaluated using $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE):

| Model | Alpha ($\alpha$) | Split | $R^2$ | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OLS Regression** | — | Train <br> Test | 0.528 <br> 0.453 | 43.48 <br> 42.79 | 53.56 <br> 53.85 |
| **RidgeCV** | 0.0944 | Train <br> Test | 0.521 <br> 0.461 | 43.89 <br> 42.99 | 53.95 <br> 53.45 |
| **LassoCV** | 0.0784 | Train <br> Test | 0.519 <br> **0.471** | 44.09 <br> **42.79** | 54.06 <br> **52.92** |

---

## V. Key Insights & Clinical Interpretability
The regularized models successfully stabilized coefficient estimations and reduced overfitting. 

### 1. Dominant Predictors
Body Mass Index (**`bmi`**) and Serum Triglycerides (**`s5`**) are consistently identified as the primary drivers of disease progression across all models. In the LassoCV model:
* **`bmi`** has a standardized coefficient of **+521.36**.
* **`s5`** has a standardized coefficient of **+498.44**.
* **`bp`** (mean blood pressure) is the third most influential predictor at **+291.56**.

### 2. Feature Sparsity & Dimensionality Reduction
The LassoCV model automatically zeroed out three features: **`age`**, **`s2`** (LDL), and **`s4`** (Cholesterol/HDL). It dropped `s2` and `s4` due to their high collinearity with `s1`, `s3`, and `s5`. This sparsity reduces clinical data collection requirements from 10 variables to 7, simplifying the clinical diagnostic process while improving test $R^2$ performance from 0.453 to 0.471.

---

## VI. Diagnostic Validation & Statistical Soundness
To verify that linear modeling assumptions were met, diagnostic tests were run on LassoCV's test residuals:
1. **Linearity and Homoscedasticity**: The residuals vs. fitted plot showed a uniform spread of residuals above and below the zero axis across the prediction range. This confirms that the relationships are linear and the error variance is stable (homoscedastic).
2. **Normality of Errors**: The normal Q-Q plot of residuals showed sample quantiles aligning with theoretical normal quantiles. This confirms that the errors are normally distributed, validating downstream hypothesis tests.

---

## VII. Strategic Recommendations for Deployability
* **Model Selection**: LassoCV is recommended for production deployment due to its superior generalization, stable parameter estimation under collinearity, and simplified feature requirements.
* **Clinical Targets**: Preventive clinical interventions should focus primarily on body weight management (`bmi`), blood pressure control (`bp`), and triglyceride management (`s5`), as these physiological metrics are the strongest predictors of disease progression.
