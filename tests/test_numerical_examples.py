from ai_computational_approaches.numerical_ml import classical_ml_examples
from ai_computational_approaches.numerical_neural import all_neural_examples
from ai_computational_approaches.numerical_symbolic import (
    knowledge_graph_example,
    reasoning_example,
)
from ai_computational_approaches.numerical_transfer_optimisation import (
    optimisation_examples,
    transfer_learning_example,
)


def test_symbolic_examples():
    knowledge = knowledge_graph_example()
    reasoning = reasoning_example()
    assert "AIModel" in knowledge["inferred_types"]
    assert knowledge["accuracy_per_ms"] > 0
    assert reasoning["bayes"]["posterior"] == 0.5
    assert reasoning["induction"]["confidence"] == 0.8


def test_classical_ml_examples():
    results = classical_ml_examples()
    assert results["dataset_summary"]["classification_rows"] == 24
    assert results["dataset_summary"]["regression_rows"] == 20
    for method in [
        "decision_tree",
        "random_forest",
        "logistic_regression",
        "knn",
        "naive_bayes",
    ]:
        assert 0.0 <= results[method]["accuracy"] <= 1.0
        assert len(results[method]["predictions"]) == 8
    assert len(results["linear_regression"]["predictions"]) == 6


def test_neural_examples():
    results = all_neural_examples()
    assert 0.0 <= results["feedforward"]["probability"] <= 1.0
    assert len(results["rnn"]["hidden_states"]) == 4
    assert len(results["lstm"]["steps"]) == 3
    assert len(results["cnn"]["feature_map"]) == 3
    assert 0.0 <= results["gan"]["probability_real"] <= 1.0
    assert len(results["bert_style"]["attention_weights"]) == 3
    assert len(results["xlnet_style"]["permutations"]) == 2


def test_transfer_learning_example():
    result = transfer_learning_example()
    assert result["source_feature_count"] == 6
    assert result["target_representation_dimensions"] == 3
    assert len(result["predictions"]) == result["test_samples"]
    assert 0.0 <= result["accuracy"] <= 1.0


def test_optimisation_examples():
    results = optimisation_examples()
    for method in [
        "genetic_algorithm",
        "evolution_strategy",
        "simulated_annealing",
        "particle_swarm",
    ]:
        assert results[method]["best_score"] >= 0.0
        assert len(results[method]["best_solution"]) == 2
