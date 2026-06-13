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

Data-driven AI learns from data. This includes traditional machine learning, deep learning, transfer learning, foundation models, and optimisation-based methods.

Examples:

- Decision trees
- Random forests
- Regression models
- KNN
- Naive Bayes
- Neural networks
- Transformers
- Transfer learning
- Large language models
- Large vision models
- Vision-language models
- Vision-language-action models

## Foundation-model families

Large model families should be classified by their inputs, outputs, and operational role rather than by parameter count alone.

| Family | Inputs | Outputs | Main role |
|---|---|---|---|
| LLM | text or tokens | text, code, structured tokens | language modelling |
| Large vision model | image or video | visual features, labels, boxes, masks | perception and representation |
| VLM | image/video plus text | captions, answers, retrieval scores, grounding | cross-modal understanding |
| VLA | vision, language, state/history | actions, trajectories, policies | embodied control |

See [`llm-large-vision-vlm-vla.md`](llm-large-vision-vlm-vla.md) for the detailed taxonomy and terminology notes.

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
| Perception | CNN, vision transformer, image recognition, feature extraction |
| Cross-modal understanding | CLIP-style alignment, captioning, VQA, grounding |
| Embodied decision and control | action policies, robot trajectories, VLA systems |
| Generative modelling | GAN, diffusion, language generation, multimodal generation |

## Computational characteristics

| Characteristic | Meaning |
|---|---|
| Data-based or knowledge-based | Whether intelligence comes from training data or explicit knowledge |
| Infrastructure-based | Hardware and platform requirements such as CPU, GPU, cloud, or edge |
| Algorithm-dependent | Loss functions, learning criteria, and optimisation behaviour |
| Multi-step or end-to-end | Whether the system uses modular stages or learns direct input-output mapping |
| Modality boundary | Whether the system operates on language, vision, multimodal data, or actions |
| Action consequence | Whether outputs remain informational or directly affect software or physical environments |

## Related code

- `src/ai_computational_approaches/knowledge.py`
- `src/ai_computational_approaches/logic_reasoning.py`
- `src/ai_computational_approaches/standard_ml.py`
- `src/ai_computational_approaches/neural_networks.py`
- `src/ai_computational_approaches/transfer_learning.py`
- `src/ai_computational_approaches/metaheuristics.py`
