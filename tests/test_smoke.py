from ai_computational_approaches.logic_reasoning import bayes_rule, modus_ponens
from ai_computational_approaches.knowledge import SimpleOntology
from ai_computational_approaches.standard_ml import decision_tree_demo
from ai_computational_approaches.metaheuristics import genetic_algorithm


def test_logic():
    assert modus_ponens(("P", "Q"), "P") == "Q"
    assert 0 <= bayes_rule(0.1, 0.8, 0.2) <= 1


def test_ontology():
    onto = SimpleOntology()
    onto.add_subclass("A", "B")
    onto.add_instance("x", "A")
    assert "B" in onto.infer_instance_types("x")


def test_standard_ml():
    assert decision_tree_demo()["accuracy"] > 0.5


def test_metaheuristic():
    assert "best_score" in genetic_algorithm(generations=5, population_size=10)
