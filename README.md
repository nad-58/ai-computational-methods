# AI Computational Methods

A practical Python-based taxonomy and implementation guide for major computational methods used in AI systems.

This repository explains AI methods from **knowledge-driven symbolic reasoning** to **data-driven machine learning**, **neural-network architectures**, **transfer learning**, and **metaheuristic optimisation**. It is designed as a clean public portfolio project with runnable Python examples.

> This public repository uses original explanations, synthetic examples, and clean diagrams. It does not include organization names, standard numbers, proprietary training slides, copyrighted course material, internal course content, or confidential information.

## Why this repository exists

AI systems are not all based on neural networks. Some rely on explicit knowledge, logic, rules, ontologies, and knowledge graphs. Others learn from data using statistical learning, traditional machine learning, deep neural networks, or optimisation algorithms.

This repository helps readers understand:

- how AI computational methods can be categorised;
- the difference between knowledge-driven and data-driven AI;
- when symbolic reasoning, machine learning, neural networks, transfer learning, or metaheuristics are useful;
- how each method works through small runnable Python examples;
- what strengths, limitations, and governance considerations apply to each family of methods.

## Visual overview

![AI taxonomy overview](docs/images/ai-taxonomy-overview.svg)

## Repository structure

```text
ai-computational-methods/
├── README.md
├── requirements.txt
├── requirements-extra.txt
├── src/ai_computational_approaches/
│   ├── knowledge.py
│   ├── logic_reasoning.py
│   ├── standard_ml.py
│   ├── neural_networks.py
│   ├── transfer_learning.py
│   └── metaheuristics.py
├── examples/
│   ├── run_all.py
│   ├── knowledge_driven_demo.py
│   ├── logic_reasoning_demo.py
│   ├── standard_ml_demo.py
│   ├── neural_network_demo.py
│   ├── transfer_learning_demo.py
│   └── metaheuristics_demo.py
├── docs/
│   ├── taxonomy.md
│   ├── knowledge-driven-approaches.md
│   ├── logic-and-reasoning.md
│   ├── data-driven-machine-learning.md
│   ├── neural-network-approaches.md
│   ├── transfer-learning.md
│   ├── metaheuristics.md
│   ├── method-comparison.md
│   └── images/
└── tests/
    └── test_smoke.py
```

## Method families covered

| Family | Methods included | Python module |
|---|---|---|
| Knowledge-driven AI | Ontology, semantic triples, knowledge graph | `knowledge.py` |
| Logic and reasoning | Deductive reasoning, inductive reasoning, hypothetical reasoning, Bayesian inference | `logic_reasoning.py` |
| Traditional ML | Decision tree, random forest, linear regression, logistic regression, KNN, Naive Bayes | `standard_ml.py` |
| Neural networks | Feedforward neural network plus architecture summaries for RNN, LSTM, CNN, GAN, BERT/XLNet-style models | `neural_networks.py` |
| Transfer learning | Reusable representation plus target classifier | `transfer_learning.py` |
| Metaheuristics | Genetic algorithm, evolution strategy, simulated annealing, particle swarm optimisation | `metaheuristics.py` |

## Quick start

```bash
git clone https://github.com/nad-58/ai-computational-methods.git
cd ai-computational-methods
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONPATH=src python examples/run_all.py
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Optional deep-learning libraries:

```bash
pip install -r requirements-extra.txt
```

## Example commands

```bash
PYTHONPATH=src python examples/knowledge_driven_demo.py
PYTHONPATH=src python examples/logic_reasoning_demo.py
PYTHONPATH=src python examples/standard_ml_demo.py
PYTHONPATH=src python examples/neural_network_demo.py
PYTHONPATH=src python examples/transfer_learning_demo.py
PYTHONPATH=src python examples/metaheuristics_demo.py
```

## Diagrams

The repository includes clean, original SVG diagrams:

| Diagram | Purpose |
|---|---|
| [`docs/images/ai-taxonomy-overview.svg`](docs/images/ai-taxonomy-overview.svg) | Overall method taxonomy |
| [`docs/images/knowledge-vs-data-driven.svg`](docs/images/knowledge-vs-data-driven.svg) | Knowledge-driven vs data-driven AI |
| [`docs/images/neural-network-family.svg`](docs/images/neural-network-family.svg) | FFNN, RNN, LSTM, CNN, GAN, BERT/XLNet-style families |
| [`docs/images/bert-vs-xlnet.svg`](docs/images/bert-vs-xlnet.svg) | Conceptual comparison of BERT-style and XLNet-style contextual language modelling |
| [`docs/images/genetic-algorithm-cycle.svg`](docs/images/genetic-algorithm-cycle.svg) | Metaheuristic optimisation cycle |

## Professional positioning

This repository demonstrates broad AI literacy across symbolic AI, traditional machine learning, neural networks, transfer learning, and optimisation. It is useful for AI engineering, AI governance, model evaluation, technical assurance, and educational portfolios.

## Disclaimer

This is an educational and professional portfolio repository. The examples are simplified and synthetic. Production AI systems require domain-specific validation, data governance, robustness testing, monitoring, security review, privacy assessment, and responsible AI governance.
