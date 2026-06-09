"""Run numerical examples for every computational method family.

From the repository root:

    PYTHONPATH=src python examples/numerical_all_methods_demo.py
"""
from __future__ import annotations

import json

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


def build_report() -> dict:
    return {
        "knowledge_driven": knowledge_graph_example(),
        "logic_and_reasoning": reasoning_example(),
        "traditional_machine_learning": classical_ml_examples(),
        "neural_networks": all_neural_examples(),
        "transfer_learning": transfer_learning_example(),
        "metaheuristics": optimisation_examples(),
    }


def main() -> None:
    report = build_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
