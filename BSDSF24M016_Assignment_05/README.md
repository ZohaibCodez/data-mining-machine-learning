
# 📓 Assignment 05 — Logistic Regression (Coursera ML Specialization Lab)

The graded logistic regression lab from the Coursera Machine Learning Specialization (Course 1, Week 3), as part of the [Data Mining & Machine Learning](../README.md) coursework. Logistic regression and its regularized variant are implemented from scratch with NumPy and applied to two classification datasets.

## 📚 Notebook

- [`Logistic_Regression.ipynb`](Logistic_Regression.ipynb) — sigmoid, cost, gradient, gradient descent, prediction, feature mapping, and regularized cost/gradient, each checked against the provided unit tests.

## 📂 Contents

- `data/ex2data1.txt` — exam scores (2 features) and admission result, 100 examples
- `data/ex2data2.txt` — two microchip test scores and accept/reject result, 118 examples
- `utils.py` — helper functions (`load_data`, `plot_data`, `map_feature`, `plot_decision_boundary`)
- `public_tests.py`, `test_utils.py` — provided autograder tests
- `images/` — figures used in the notebook text

## 🧮 Approach

1. **Sigmoid:** `g(z) = 1 / (1 + e^(−z))`, applied element-wise
2. **Cost:** `J(w,b) = (1/m) Σ [ −y·log(f) − (1−y)·log(1−f) ]`
3. **Gradient:** `∂J/∂w_j = (1/m) Σ (f − y)·x_j`, `∂J/∂b = (1/m) Σ (f − y)`
4. **Gradient descent:** batch updates `w ← w − α·∂J/∂w`, `b ← b − α·∂J/∂b`
5. **Predict:** class 1 when `f(x) ≥ 0.5`, otherwise 0
6. **Regularization:** add `(λ / 2m) Σ w_j²` to the cost and `(λ / m)·w_j` to each weight gradient
7. **Feature mapping:** the two microchip scores are expanded to all polynomial terms up to degree 6 (27 features), giving a non-linear decision boundary

## 📊 Results Summary

| Dataset | Features | Settings | Final Cost | Train Accuracy |
|---------|----------|----------|------------|----------------|
| Exam scores (`ex2data1`) | 2 | α = 0.001, 10,000 iterations | 0.30 | 92.0% |
| Microchips (`ex2data2`) | 27 (degree-6 mapping) | α = 0.01, λ = 0.01, 10,000 iterations | 0.45 | 82.2% |

Checks along the way: initial cost with `w = 0, b = 0` is `0.693`, and the regularized cost with the provided test values is `0.662`. All six unit tests pass.

## ▶️ Running

From the repo root:

```bash
jupyter notebook "BSDSF24M016_Assignment_05/Logistic_Regression.ipynb"
```

The notebook reads the data from `data/`, so run it from within this folder or keep the folder structure intact. On Google Colab, upload `utils.py`, `public_tests.py`, `test_utils.py` and the `data/` folder first.
