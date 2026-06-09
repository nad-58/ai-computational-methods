# Numerical Examples for All AI Computational Methods

This tutorial provides small, deterministic numerical examples for every method family named in the repository. The examples are intentionally compact so readers can inspect the input data, intermediate calculations, outputs, and evaluation metrics.

## Run all numerical examples

From the repository root:

```bash
PYTHONPATH=src python examples/numerical_all_methods_demo.py
```

Run the numerical tests:

```bash
PYTHONPATH=src pytest tests/test_numerical_examples.py -q
```

## Data files

| File | Shape | Purpose |
|---|---:|---|
| `data/numerical_classification_dataset.csv` | 24 rows × 4 features | Classification examples |
| `data/regression_example.csv` | 12 rows × 3 features | Linear-regression illustration |
| `data/sequence_example.csv` | 4 time steps | RNN sequence illustration |
| `data/image_matrix_example.csv` | 4 × 4 matrix | CNN convolution illustration |
| `data/token_embeddings_example.csv` | 3 tokens × 2 dimensions | BERT-style attention illustration |

The runnable ML functions generate deterministic synthetic datasets with fixed random seed `7`. The CSV files provide inspectable teaching versions of the same data types.

---

## 1. Knowledge-driven AI

### Input

A model has the following numerical properties:

```text
accuracy = 92%
latency = 38 ms
risk score = 7
```

The ontology states:

```text
HighRiskModel -> AIModel -> SoftwareAsset
model_17 is a HighRiskModel
```

### Flow

```text
facts -> ontology and knowledge graph -> inference and query -> derived indicator
```

The ontology infers three types for `model_17`:

```text
HighRiskModel, AIModel, SoftwareAsset
```

A simple derived efficiency indicator is:

```text
accuracy per millisecond = 92 / 38 = 2.4211
```

This example demonstrates ontology inheritance, instance inference, knowledge-graph querying, and a derived numerical value.

---

## 2. Logic and reasoning

### Deductive reasoning

Given:

```text
If temperature > 80, then alarm = true
temperature > 80
```

Modus ponens infers:

```text
alarm = true
```

Given:

```text
If motor is on, current is positive
current is not positive
```

Modus tollens infers:

```text
motor is not on
```

### Inductive reasoning

Observed labels:

```text
[1, 1, 1, 0, 1]
```

The majority label is `1` and its empirical confidence is:

```text
4 / 5 = 0.8
```

### Hypothesis testing

Observed values:

```text
[9.7, 9.8, 9.9]
```

Hypothesis:

```text
predicted value = 9.8
tolerance = ±0.2
```

All three observations are within tolerance, so the simple hypothesis is supported.

### Bayesian inference

Given:

```text
prior = 0.10
likelihood = 0.90
evidence probability = 0.18
```

The posterior is:

```text
posterior = likelihood × prior / evidence
          = 0.90 × 0.10 / 0.18
          = 0.50
```

---

## 3. Traditional machine learning

The runnable example creates:

```text
Classification: 24 rows, 4 features, 2 classes
Regression: 20 rows, 3 features, continuous target
```

### Classification flow

```text
24 synthetic rows
-> first 16 training rows
-> final 8 test rows
-> fit classifier
-> predict 8 labels
-> calculate accuracy
```

The same fixed dataset is used for:

- Decision tree
- Random forest
- Logistic regression
- K-nearest neighbours, with `k = 3`
- Gaussian Naive Bayes

For every model, the output reports:

```text
training samples
test samples
first test feature vector
true labels
predicted labels
accuracy
```

### Linear regression flow

```text
20 rows × 3 features
-> 14 training rows
-> 6 test rows
-> fit y = b0 + b1x1 + b2x2 + b3x3
-> predict continuous values
-> calculate MAE and R²
```

The output includes learned coefficients, intercept, test targets, predictions, mean absolute error, and R².

---

## 4. Feedforward neural network

Input vector:

```text
x = [0.6, -0.2]
```

First-layer weights:

```text
W1 = [[0.5, -0.3],
      [0.8,  0.2]]

b1 = [0.1, -0.1]
```

The calculation is:

```text
hidden linear = xW1 + b1
hidden activation = tanh(hidden linear)
output linear = hiddenW2 + b2
probability = sigmoid(output linear)
```

The example reports every intermediate vector and the final binary prediction.

---

## 5. Recurrent neural network

Sequence:

```text
[0.2, 0.5, -0.1, 0.7]
```

Scalar recurrence:

```text
h_t = tanh(0.8x_t + 0.4h_(t-1) + 0.1)
```

Flow:

```text
current input + previous hidden state -> tanh -> new hidden state
```

The example reports all four hidden states and the final sequence representation.

---

## 6. LSTM

Sequence:

```text
[0.4, -0.2, 0.6]
```

At every step the example calculates:

```text
forget gate
input gate
output gate
candidate memory
cell state
hidden state
```

Flow:

```text
input -> gates -> updated cell memory -> hidden output
```

This is a one-cell educational LSTM, not a trained production model.

---

## 7. Convolutional neural network

Input image:

```text
0 0 1 1
0 1 1 0
1 1 0 0
1 0 0 0
```

Kernel:

```text
 1  0
 0 -1
```

The 2 × 2 kernel slides across the 4 × 4 image. At each location:

```text
activation = sum(image patch × kernel)
```

The result is a 3 × 3 feature map. The example reports the full feature map and maximum activation.

---

## 8. GAN

Noise vector:

```text
z = [0.3, -0.5]
```

Flow:

```text
noise -> generator matrix -> tanh synthetic sample
-> discriminator dot product -> sigmoid probability
-> generator adversarial loss
```

The example reports the generated sample, discriminator logit, probability that the sample is real, and generator loss.

---

## 9. BERT-style attention

Tokens and embeddings:

```text
AI   -> [1.0, 0.0]
uses -> [0.5, 0.5]
data -> [0.0, 1.0]
```

The example uses the same small vectors as queries, keys, and values:

```text
scores = embeddings × embeddingsᵀ / sqrt(2)
attention = softmax(scores)
context = attention × embeddings
```

It reports the 3 × 3 score matrix, attention matrix, and contextual embeddings.

---

## 10. XLNet-style permutation modelling

Token values:

```text
A = 1, B = 2, C = 3
```

Two factorisation orders are compared:

```text
A -> B -> C
C -> A -> B
```

For each token:

```text
prediction score = 0.6 × accumulated context + 0.4 × token value
```

The output shows how changing the permutation changes the available autoregressive context.

---

## 11. Transfer learning

Synthetic target task:

```text
40 rows × 6 raw features
28 training rows
12 test rows
```

Flow:

```text
raw target features
-> standardisation
-> reused PCA representation with 3 dimensions
-> logistic-regression target classifier
-> predictions and accuracy
```

The output includes the PCA explained-variance ratios and the first raw and transformed target vectors.

---

## 12. Metaheuristics

All four methods minimise the same numerical objective:

```text
f(x1, x2) = x1² + x2²
```

The global optimum is:

```text
x = [0, 0]
f(x) = 0
```

The following algorithms are run with fixed seed `7`:

- Genetic algorithm: population 20, generations 30
- Evolution strategy: 30 generations
- Simulated annealing: 150 steps
- Particle swarm optimisation: 15 particles, 30 iterations

Flow:

```text
candidate vectors -> objective scores -> algorithm-specific update -> improved vector
```

Each result reports the best two-dimensional solution and its final objective score.

---

## Interpretation

These examples are deliberately small. Their purpose is to expose the numerical flow, not to claim benchmark performance. Real systems require larger representative datasets, repeated runs, uncertainty analysis, tuning, robustness checks, and independent validation.
