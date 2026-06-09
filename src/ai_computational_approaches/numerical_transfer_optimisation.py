"""Numerical examples for transfer learning and metaheuristic optimisation."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ai_computational_approaches.metaheuristics import (
    evolution_strategy,
    genetic_algorithm,
    particle_swarm_optimisation,
    simulated_annealing,
)


def _round(values, digits: int = 4):
    return np.asarray(values).round(digits).tolist()


def transfer_learning_example(seed: int = 7) -> dict:
    """Reuse a learned representation, then train a small target classifier."""
    X, y = make_classification(
        n_samples=40,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=seed,
    )
    train_X, test_X = X[:28], X[28:]
    train_y, test_y = y[:28], y[28:]

    scaler = StandardScaler().fit(train_X)
    train_scaled = scaler.transform(train_X)
    test_scaled = scaler.transform(test_X)

    representation = PCA(n_components=3, random_state=seed).fit(train_scaled)
    train_representation = representation.transform(train_scaled)
    test_representation = representation.transform(test_scaled)

    classifier = LogisticRegression(max_iter=1000).fit(train_representation, train_y)
    predictions = classifier.predict(test_representation)

    return {
        "source_feature_count": 6,
        "target_representation_dimensions": 3,
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "explained_variance_ratio": _round(representation.explained_variance_ratio_),
        "first_raw_target_vector": _round(test_X[0]),
        "first_reused_representation": _round(test_representation[0]),
        "targets": test_y.tolist(),
        "predictions": predictions.tolist(),
        "accuracy": round(float(accuracy_score(test_y, predictions)), 4),
        "flow": "raw target data -> reused scaler/PCA -> target classifier -> prediction",
    }


def optimisation_examples() -> dict:
    """Optimise f(x1,x2)=x1^2+x2^2 with four algorithms."""
    runs = {
        "genetic_algorithm": genetic_algorithm(
            dimensions=2,
            population_size=20,
            generations=30,
            seed=7,
        ),
        "evolution_strategy": evolution_strategy(
            dimensions=2,
            generations=30,
            seed=7,
        ),
        "simulated_annealing": simulated_annealing(
            dimensions=2,
            steps=150,
            seed=7,
        ),
        "particle_swarm": particle_swarm_optimisation(
            dimensions=2,
            particles=15,
            iterations=30,
            seed=7,
        ),
    }
    results = {}
    for name, result in runs.items():
        results[name] = {
            "best_solution": _round(result["best_solution"], 6),
            "best_score": round(float(result["best_score"]), 8),
        }
    results["objective"] = "minimise f(x1,x2)=x1^2+x2^2; optimum [0,0], score 0"
    results["flow"] = "candidate vectors -> objective score -> algorithm update -> improved vector"
    return results
