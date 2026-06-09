"""Numerical examples for knowledge-driven and reasoning methods."""
from ai_computational_approaches.knowledge import SimpleKnowledgeGraph, SimpleOntology
from ai_computational_approaches.logic_reasoning import (
    bayes_rule,
    inductive_majority_rule,
    modus_ponens,
    modus_tollens,
    test_hypothesis,
    transitive_inference,
)


def knowledge_graph_example() -> dict:
    ontology = SimpleOntology()
    ontology.add_subclass("HighRiskModel", "AIModel")
    ontology.add_subclass("AIModel", "SoftwareAsset")
    ontology.add_instance("model_17", "HighRiskModel")

    graph = SimpleKnowledgeGraph()
    graph.add("model_17", "accuracy_percent", "92")
    graph.add("model_17", "latency_ms", "38")
    graph.add("model_17", "risk_score", "7")
    attributes = {p: float(v) for p, v in graph.neighbours("model_17")}
    return {
        "inferred_types": sorted(ontology.infer_instance_types("model_17")),
        "numeric_attributes": attributes,
        "accuracy_per_ms": round(attributes["accuracy_percent"] / attributes["latency_ms"], 4),
        "flow": "facts -> ontology/graph -> inference/query -> numerical indicator",
    }


def reasoning_example() -> dict:
    labels = [1, 1, 1, 0, 1]
    majority, confidence = inductive_majority_rule(labels)
    return {
        "modus_ponens": modus_ponens(("temperature_gt_80", "alarm"), "temperature_gt_80"),
        "modus_tollens": modus_tollens(("motor_on", "current_positive"), "not current_positive"),
        "transitive": transitive_inference(("A", "B"), ("B", "C")),
        "induction": {"samples": labels, "majority": majority, "confidence": confidence},
        "hypothesis": test_hypothesis([9.7, 9.8, 9.9], 9.8, 0.2),
        "bayes": {
            "prior": 0.10,
            "likelihood": 0.90,
            "evidence": 0.18,
            "posterior": round(bayes_rule(0.10, 0.90, 0.18), 4),
        },
        "flow": "observations/rules -> reasoning operation -> conclusion",
    }
