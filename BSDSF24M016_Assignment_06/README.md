
# 📓 Assignment 06 — Naive Bayes Classifier (Digit 4 vs Digit 2)

An optical character recognition system that distinguishes images of the digit 4 from the digit 2, using a Naive Bayes classifier implemented from scratch with NumPy, as part of the [Data Mining & Machine Learning](../README.md) coursework.

## 🎯 Goal

Classify 16×16 binary images (256 pixel features) of the digits 2 and 4, and report the overall and per-class accuracy on both the training and test sets.

## 📚 Notebook

- [`Naive_Bayes.ipynb`](Naive_Bayes.ipynb) — data loading, image visualization, Bernoulli Naive Bayes with Laplace smoothing, and overall and per-class accuracy.

## 📄 Data Files

Derived from the [STPRtool](http://cmp.felk.cvut.cz/cmp/software/stprtool/index.html) dataset.

- [`trainX.txt`](trainX.txt), [`trainY.txt`](trainY.txt) — 500 training images (250 per class) and labels
- [`testX.txt`](testX.txt), [`testY.txt`](testY.txt) — 100 test images (50 per class) and labels
- `Naive bayes classifier.doc` — assignment statement

## 🧮 Approach

1. Load each image as 256 binary pixels (reshaped to 16×16 with `order='F'` for display)
2. Estimate the class priors `P(c) = N_c / N`
3. Estimate per-pixel likelihoods with Laplace smoothing: `θ_cj = (Σ x_ij + 1) / (N_c + 2)`
4. Predict `argmax_c  log P(c) + Σ_j [ x_j·log θ_cj + (1 − x_j)·log(1 − θ_cj) ]`, computed in log space to avoid underflow

## 📊 Results Summary

| Class | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| 2 | 98.40% | 100.00% |
| 4 | 99.20% | 100.00% |
| **Overall** | **98.80%** | **100.00%** |

**Takeaway:** Despite the unrealistic assumption that pixels are independent given the class, Naive Bayes separates the two digits almost perfectly. Only 6 of the 500 training images are misclassified and none of the 100 test images, so there is no sign of overfitting.

## ▶️ Running

From the repo root:

```bash
jupyter notebook "BSDSF24M016_Assignment_06/Naive_Bayes.ipynb"
```

Run it from within the assignment folder (or keep the folder intact), since the data files are read by relative path.
