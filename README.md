# Comparative Implementation of Lasso and Ridge Regression using Subgradient Optimization Techniques

## 📌 Project Overview

This project focuses on implementing and comparing three regression models:

* Linear Regression (Baseline)
* Ridge Regression (L2 Regularization)
* Lasso Regression (L1 Regularization using Subgradient Method)

The objective is to understand how regularization improves model performance and reduces overfitting.

---

## 📊 Dataset

The project uses the **Diabetes Dataset** from scikit-learn.

* Number of samples: 442
* Number of features: 10
* Target: Disease progression after one year

---

## ⚙️ Techniques Used

* Linear Regression using Gradient Descent
* Ridge Regression (L2 Regularization)
* Lasso Regression (Subgradient Optimization)
* Feature Scaling (Standardization)
* Train-Test Split

---

## 📈 Evaluation Metrics

The models are evaluated using:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 📊 Results

| Model  | MSE     | RMSE  | R² Score |
| ------ | ------- | ----- | -------- |
| Linear | 2884.92 | 53.71 | 0.46     |
| Ridge  | 3153.34 | 56.15 | 0.40     |
| Lasso  | 2882.01 | 53.68 | 0.46     |

---

## 📉 Visualization

A bar chart is used to compare the R² scores of Linear, Ridge, and Lasso regression models.

---

## 🚀 How to Run the Project

1. Clone the repository:

```
git clone https://github.com/likithaharish/lasso-ridge-regression.git
```

2. Navigate to the project folder:

```
cd lasso-ridge-regression
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the script:

```
python main.py
```

---

## 🧠 Key Insights

* Linear and Lasso Regression performed similarly
* Ridge Regression showed slightly lower performance due to stronger regularization
* Feature scaling significantly improved model performance
* Lasso helps in feature selection

---

## 📌 Conclusion

This project demonstrates the importance of regularization in regression models. Lasso Regression helps in feature selection, while Ridge Regression improves model stability. Subgradient optimization is used to handle non-differentiability in Lasso.

---


