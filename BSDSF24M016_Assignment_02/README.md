# 📓 Assignment 02 — Regression Models From Scratch

Polynomial regression implemented from first principles using the **normal equation** (closed-form matrix solution), as part of the [Data Mining & Machine Learning](../README.md) coursework.

## 🎯 Goal

Fit and compare regression models of increasing complexity — linear, quadratic, cubic, and degree 4–6 polynomials — to a single-feature dataset (`X` → `R`), using only NumPy matrix operations (no `sklearn`, no `np.polyfit`, no for-loops over the data).

## 📚 Notebook

- [`Regresion.ipynb`](Regresion.ipynb) — full pipeline: load data → build polynomial design matrix → solve `Θ = (XᵀX)⁻¹Xᵀy` → predict → compute MSE → plot fit → repeat for degrees 1 through 6 → compare results.

## 📄 Data Files

- [`trainRegression.csv`](trainRegression.csv) — 283 training points (`X`, `R`)
- [`testRegression.csv`](testRegression.csv) — 32 held-out test points (`X`, `R`)

## 🧮 Approach

Each model is fit by:
1. Building a design matrix whose columns are powers of `X` (`1, X, X², ..., Xⁿ`)
2. Solving the normal equation `Θ = (XᵀX)⁻¹Xᵀy` for the coefficients
3. Predicting on train/test data via matrix multiplication (`X · Θ`)
4. Scoring with Mean Square Error: `MSE = mean((y_pred - y_actual)²)`
5. Plotting actual vs. fitted values

## 📊 Results Summary

| Degree | Train MSE | Test MSE |
|--------|-----------|----------|
| 1 (Linear) | 0.299 | 0.316 |
| 2 (Quadratic) | 0.288 | 0.326 |
| 3 (Cubic) | 0.050 | 0.052 |
| 4 | 0.044 | 0.050 |
| 5 | 0.039 | 0.044 |
| 6 | 0.038 | 0.045 |

**Takeaway:** Degrees 1–2 underfit (high error on both train and test). Error drops sharply at degree 3, showing the true relationship has cubic-like structure. Degree 5 gives the best test performance overall, while degree 6 shows early signs of overfitting — training error plateaus but test error ticks back up.

## ▶️ Running

From the repo root:

```bash
jupyter notebook "BSDSF24M016_Assignment_02/Regresion.ipynb"
```
