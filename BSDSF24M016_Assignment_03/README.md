# 📓 Assignment 03 — Regression Models Using Gradient Descent

Polynomial regression (linear through degree 6) trained from scratch with batch gradient descent, as part of the [Data Mining & Machine Learning](../README.md) coursework.

## 🎯 Goal

Replace the closed-form Normal Equation from Assignment 02 with an iterative gradient descent solver, derive the weight update equations for each model, and compare the two approaches on train and test MSE.

## 📚 Notebook

- [`Regression_GradientDescent.ipynb`](Regression_GradientDescent.ipynb) — data loading, feature construction and normalisation, gradient descent for degrees 1–6, cost history plots, predictions, MSE, and the final comparison.

## 📄 Report

- [`BSDSF24M016_Assignment_03_Report.pdf`](BSDSF24M016_Assignment_03_Report.pdf) — weight update derivations, results tables, GD vs Normal Equation comparison, convergence and overfitting analysis.

## 📄 Data Files

- [`trainRegression.csv`](trainRegression.csv) — 283 training points (`X`, `R`)
- [`testRegression.csv`](testRegression.csv) — 32 held-out test points (`X`, `R`)

## 🧮 Approach

1. Build the design matrix `X` whose columns are powers of `x` (z-score normalised for degrees 3–6)
2. Start from `θ = 0` and repeat `θ ← θ − α·(1/m)·Xᵀ(Xθ − y)` for 1000 iterations, with `α = 0.01`
3. Record the cost `J(θ) = (1/2m) Σ(h − y)²` at each iteration
4. Predict on the test set and report MSE using the same `(1/2m)` convention

## 📊 Results Summary

| Degree | GD Train MSE | GD Test MSE | Normal Eq. Test MSE |
|--------|--------------|-------------|---------------------|
| 1 (Linear) | 0.1514 | 0.1571 | 0.1580 |
| 2 (Quadratic) | 0.1619 | 0.1610 | 0.1630 |
| 3 (Cubic) | 0.1365 | 0.1485 | 0.0258 |
| 4 | 0.1151 | 0.1332 | 0.0250 |
| 5 | 0.0980 | 0.1192 | 0.0221 |
| 6 | 0.0871 | 0.1086 | 0.0223 |

**Takeaway:** For linear and quadratic models, gradient descent matches the Normal Equation within 0.002. For degrees 3–6, the cost is still falling at iteration 1000, so the models have not converged and their error is about five to six times the closed-form value. Gradient descent needs a larger learning rate or more iterations to reach the minimum.

## ▶️ Running

From the repo root:

```bash
jupyter notebook "BSDSF24M016_Assignment_03/Regression_GradientDescent.ipynb"
```
