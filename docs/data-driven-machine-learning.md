# Data-Driven Machine Learning

Data-driven machine learning methods learn patterns from data rather than relying only on explicit rules.

## Methods covered

| Method | Task type | Key idea |
|---|---|---|
| Decision tree | Classification / regression | Recursive feature-based partitioning |
| Random forest | Classification / regression | Ensemble of decision trees with bootstrap sampling |
| Linear regression | Regression | Linear relationship between predictors and target |
| Logistic regression | Classification | Log-odds model for probability of class membership |
| K-nearest neighbour | Classification / regression | Predict using nearby training samples |
| Naive Bayes | Classification | Bayesian classifier with conditional independence assumption |

## Practical considerations

- Decision trees are interpretable but can overfit.
- Random forests improve robustness but are less transparent than a single tree.
- Linear and logistic regression are strong baselines and often easy to explain.
- KNN is simple but can be slow at inference time for large datasets.
- Naive Bayes is efficient but depends on a strong independence assumption.

## Related code

```bash
PYTHONPATH=src python examples/standard_ml_demo.py
```

Main module:

```text
src/ai_computational_approaches/standard_ml.py
```
