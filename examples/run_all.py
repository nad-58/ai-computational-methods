from ai_computational_approaches.knowledge import demo as knowledge_demo
from ai_computational_approaches.logic_reasoning import demo as logic_demo
from ai_computational_approaches.standard_ml import run_all_standard_ml
from ai_computational_approaches.neural_networks import architecture_summary, feedforward_demo
from ai_computational_approaches.transfer_learning import transfer_learning_demo
from ai_computational_approaches.metaheuristics import run_all_metaheuristics


def main() -> None:
    print("\nKnowledge-driven approaches")
    print(knowledge_demo())

    print("\nLogic and reasoning")
    print(logic_demo())

    print("\nStandard machine learning")
    for item in run_all_standard_ml():
        print(item)

    print("\nNeural-network examples")
    print(feedforward_demo())
    print(architecture_summary())

    print("\nTransfer learning")
    print(transfer_learning_demo())

    print("\nMetaheuristics")
    for item in run_all_metaheuristics():
        print(item)


if __name__ == "__main__":
    main()
