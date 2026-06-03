# Method Comparison

This table compares major computational method families at a practical level.

| Approach | Best for | Explainability | Data need | Compute need | Main limitation |
|---|---|---:|---:|---:|---|
| Ontology / rules | Domain logic and explicit reasoning | High | Low | Low | Manual knowledge engineering |
| Knowledge graph | Relationship-heavy knowledge | Medium-high | Medium | Medium | Graph construction and maintenance |
| Decision tree | Transparent classification | High | Medium | Low | Overfitting if not controlled |
| Random forest | Robust tabular modelling | Medium | Medium | Medium | Less transparent than one tree |
| Linear regression | Continuous prediction baseline | High | Low-medium | Low | Assumes linear relationships |
| Logistic regression | Binary classification baseline | High | Low-medium | Low | Linear decision boundary |
| KNN | Simple non-parametric baseline | Medium | Medium | High at inference | Slow for large/high-dimensional data |
| Naive Bayes | Fast probabilistic classification | Medium-high | Low-medium | Low | Conditional independence assumption |
| FFNN | General nonlinear mapping | Low-medium | Medium-high | Medium-high | Overfitting and tuning complexity |
| RNN / LSTM | Sequential data | Low-medium | High | High | Training complexity and gradient issues |
| CNN | Spatial/image data | Low-medium | High | High | Requires data and hardware |
| GAN | Synthetic data generation | Low | High | High | Training instability |
| BERT / XLNet-style | Contextual language representation | Low-medium | High pretraining / lower fine-tuning | High | Large compute and careful evaluation needed |
| Transfer learning | Limited target data | Medium | Lower target need | Medium-high | Source-target mismatch |
| Metaheuristics | Complex optimisation | Medium | Depends | Medium-high | No guarantee of global optimum |

## Practical selection guidance

- Use **knowledge-driven methods** when domain logic, traceability, or explicit rules matter.
- Use **traditional machine learning** for tabular data, baselines, explainability, and efficient modelling.
- Use **neural networks** for high-dimensional, spatial, temporal, language, or representation-learning tasks.
- Use **transfer learning** when target data is limited but related source knowledge is available.
- Use **metaheuristics** when the optimisation landscape is complex, non-differentiable, or difficult to model analytically.
