# Taxonomy of AI Computational Methods

This document gives a clean, standard-number-free taxonomy for AI computational methods.

## Two main streams

### Knowledge-driven methods

Knowledge-driven AI represents information as symbols, rules, ontologies, semantic triples, and knowledge graphs. These systems are useful when human expertise, explicit decision rules, traceability, or domain knowledge are central.

Examples:

- Expert rules
- Ontologies
- Knowledge graphs
- Semantic triples
- Deductive reasoning
- Rule-based inference

### Data-driven methods

Data-driven AI learns from data. This includes traditional machine learning, deep learning, transfer learning, and optimisation-based methods.

Examples:

- Decision trees
- Random forests
- Regression models
- KNN
- Naive Bayes
- Neural networks
- Transformers
- Transfer learning

## Purpose-based categories

AI computational methods can also be grouped by purpose:

| Purpose | Examples |
|---|---|
| Search and optimisation | heuristic search, genetic algorithm, simulated annealing |
| Logic and reasoning | deductive reasoning, inductive reasoning, Bayesian inference |
| Knowledge representation | ontology, knowledge graph, semantic representation |
| Learning from examples | supervised learning, classification, regression |
| Unsupervised discovery | clustering, dimensionality reduction |
| Sequential modelling | RNN, LSTM, transformer-style models |
| Perception | CNN, image recognition, feature extraction |
| Generative modelling | GAN, generative neural networks |

## Computational characteristics

| Characteristic | Meaning |
|---|---|
| Data-based or knowledge-based | Whether intelligence comes from training data or explicit knowledge |
| Infrastructure-based | Hardware and platform requirements such as CPU, GPU, cloud, or edge |
| Algorithm-dependent | Loss functions, learning criteria, and optimisation behaviour |
| Multi-step or end-to-end | Whether the system uses modular stages or learns direct input-output mapping |

## Related code

- `src/ai_computational_approaches/knowledge.py`
- `src/ai_computational_approaches/logic_reasoning.py`
- `src/ai_computational_approaches/standard_ml.py`
- `src/ai_computational_approaches/neural_networks.py`
- `src/ai_computational_approaches/transfer_learning.py`
- `src/ai_computational_approaches/metaheuristics.py`
