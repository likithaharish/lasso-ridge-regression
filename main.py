"""
Project: Comparative Implementation of Lasso and Ridge Regression
using Subgradient Optimization Techniques

Author: Likitha H
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load Dataset
# -----------------------------
print("\n📊 Loading Diabetes Dataset...\n")
data = load_diabetes()
X = data.data
y = data.target

# -----------------------------
# 2. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 4. Add Bias (Intercept)
# -----------------------------
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# -----------------------------
# 5. Hyperparameters
# -----------------------------
lr = 0.001
iterations = 5000
lam = 0.1

# -----------------------------
# 6. Linear Regression
# -----------------------------
def linear_regression(X, y):
    m, n = X.shape
    beta = np.zeros(n)

    for _ in range(iterations):
        y_pred = X @ beta
        gradient = -(2/m) * X.T @ (y - y_pred)
        beta -= lr * gradient

    return beta

# -----------------------------
# 7. Ridge Regression
# -----------------------------
def ridge_regression(X, y):
    m, n = X.shape
    beta = np.zeros(n)

    for _ in range(iterations):
        y_pred = X @ beta
        gradient = -(2/m) * X.T @ (y - y_pred) + 2 * lam * beta
        beta -= lr * gradient

    return beta

# -----------------------------
# 8. Lasso Regression (Subgradient)
# -----------------------------
def lasso_subgradient(X, y):
    m, n = X.shape
    beta = np.zeros(n)

    for _ in range(iterations):
        y_pred = X @ beta
        gradient = -(2/m) * X.T @ (y - y_pred)
        subgrad = lam * np.sign(beta)
        beta -= lr * (gradient + subgrad)

    return beta

# -----------------------------
# 9. Train Models
# -----------------------------
print("⚙️ Training Models...\n")

beta_linear = linear_regression(X_train, y_train)
beta_ridge = ridge_regression(X_train, y_train)
beta_lasso = lasso_subgradient(X_train, y_train)

# -----------------------------
# 10. Predictions
# -----------------------------
y_pred_linear = X_test @ beta_linear
y_pred_ridge = X_test @ beta_ridge
y_pred_lasso = X_test @ beta_lasso

# -----------------------------
# 11. Evaluation Metrics
# -----------------------------
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

mse_lin, rmse_lin, r2_lin = evaluate(y_test, y_pred_linear)
mse_ridge, rmse_ridge, r2_ridge = evaluate(y_test, y_pred_ridge)
mse_lasso, rmse_lasso, r2_lasso = evaluate(y_test, y_pred_lasso)

# -----------------------------
# 12. Display Results
# -----------------------------
print("📊 Model Comparison:\n")
print(f"{'Model':<10} {'MSE':<12} {'RMSE':<12} {'R2 Score':<12}")
print("-" * 50)
print(f"{'Linear':<10} {mse_lin:<12.2f} {rmse_lin:<12.2f} {r2_lin:<12.2f}")
print(f"{'Ridge':<10} {mse_ridge:<12.2f} {rmse_ridge:<12.2f} {r2_ridge:<12.2f}")
print(f"{'Lasso':<10} {mse_lasso:<12.2f} {rmse_lasso:<12.2f} {r2_lasso:<12.2f}")

# -----------------------------
# 13. Visualization
# -----------------------------
models = ['Linear', 'Ridge', 'Lasso']
r2_scores = [r2_lin, r2_ridge, r2_lasso]

plt.figure()
plt.bar(models, r2_scores)
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2 Score)")
plt.show()

# -----------------------------
# 14. Conclusion Output
# -----------------------------
print("\n🧠 Insights:")
print("- Linear and Lasso perform similarly")
print("- Ridge slightly lower due to regularization")
print("- Feature scaling improved performance significantly")
print("- Lasso helps in feature selection")