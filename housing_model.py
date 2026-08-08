import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------
# Step 1: Load and Explore the Dataset
# ---------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("--- Dataset Information ---")
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())

# Features (X) and Target (y)
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']  # Target: Median house value in $100,000s

# ---------------------------------------------------------
# Step 2: Train-Test Split
# ---------------------------------------------------------
# Split 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# Step 3: Feature Scaling
# ---------------------------------------------------------
# Multiple Linear Regression performs better when features are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Step 4: Model Training
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# Step 5: Model Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE)  : ${mae * 100000:,.2f}  ({mae:.4f} units)")
print(f"Root Mean Squared Error (RMSE): ${rmse * 100000:,.2f}  ({rmse:.4f} units)")
print(f"R-squared (R² Score)       : {r2:.4f}")

# ---------------------------------------------------------
# Step 6: Feature Coefficients Analysis
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\n--- Feature Coefficients ---")
print(f"Intercept: {model.intercept_:.4f}")
print(coef_df.to_string(index=False))

# ---------------------------------------------------------
# Step 7: Visualizing Model Performance
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs Predicted Prices
axes[0].scatter(y_test, y_pred, alpha=0.3, color='crimson')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0].set_xlabel("Actual Price ($100k)")
axes[0].set_ylabel("Predicted Price ($100k)")
axes[0].set_title("Actual vs. Predicted House Prices")

# Plot 2: Residual Plot (Error Distribution)
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.3, color='teal')
axes[1].axhline(y=0, color='k', linestyle='--', lw=2)
axes[1].set_xlabel("Predicted Price ($100k)")
axes[1].set_ylabel("Residuals (Error)")
axes[1].set_title("Residual Plot (Check for Heteroscedasticity)")

plt.tight_layout()
plt.show()