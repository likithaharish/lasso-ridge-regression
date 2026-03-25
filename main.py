import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Generate Synthetic Dataset
# -----------------------------
X, y, true_coef = make_regression(
    n_samples=200,
    n_features=20,
    n_informative=5,
    noise=10,
    coef=True,
    random_state=42
)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. LASSO Using Subgradient
# -----------------------------
def lasso_subgradient(X, y, lambda_, lr=0.01, iterations=500):
    n, m = X.shape
    w = np.zeros(m)
    losses = []

    for _ in range(iterations):
        # Prediction
        y_pred = X @ w

        # MSE Gradient
        grad_mse = (1/n) * X.T @ (y_pred - y)

        # Subgradient of L1
        subgrad = lambda_ * np.sign(w)

        # Update rule
        w = w - lr * (grad_mse + subgrad)

        # Compute loss
        loss = (1/(2*n)) * np.sum((y_pred - y)**2) + lambda_ * np.sum(np.abs(w))
        losses.append(loss)

    return w, losses


# -----------------------------
# 3. Ridge Regression (Comparison)
# -----------------------------
def ridge_regression(X, y, lambda_, lr=0.01, iterations=500):
    n, m = X.shape
    w = np.zeros(m)
    losses = []

    for _ in range(iterations):
        y_pred = X @ w

        grad = (1/n) * X.T @ (y_pred - y) + 2 * lambda_ * w

        w = w - lr * grad

        loss = (1/(2*n)) * np.sum((y_pred - y)**2) + lambda_ * np.sum(w**2)
        losses.append(loss)

    return w, losses


# -----------------------------
# 4. Train Models
# -----------------------------
lambda_ = 0.1

lasso_w, lasso_losses = lasso_subgradient(X_train, y_train, lambda_)
ridge_w, ridge_losses = ridge_regression(X_train, y_train, lambda_)


# -----------------------------
# 5. Evaluation
# -----------------------------
def mean_squared_error(X, y, w):
    y_pred = X @ w
    return np.mean((y - y_pred)**2)

lasso_mse = mean_squared_error(X_test, y_test, lasso_w)
ridge_mse = mean_squared_error(X_test, y_test, ridge_w)

print("LASSO Test MSE:", lasso_mse)
print("Ridge Test MSE:", ridge_mse)

print("\nNumber of zero weights in LASSO:",
      np.sum(np.isclose(lasso_w, 0, atol=1e-3)))

print("Number of zero weights in Ridge:",
      np.sum(np.isclose(ridge_w, 0, atol=1e-3)))


# -----------------------------
# 6. Plot Convergence
# -----------------------------
plt.figure()
plt.plot(lasso_losses)
plt.title("LASSO Convergence (Subgradient Method)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

plt.figure()
plt.plot(ridge_losses)
plt.title("Ridge Convergence")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()