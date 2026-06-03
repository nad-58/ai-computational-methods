"""Knowledge-driven AI examples: ontology, knowledge graph, and semantic triples."""
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str


class SimpleOntology:
    """Minimal ontology with subclass and instance inference."""

    def __init__(self) -> None:
        self.subclass_of: Dict[str, Set[str]] = {}
        self.instance_of: Dict[str, Set[str]] = {}

    def add_subclass(self, child: str, parent: str) -> None:
        self.subclass_of.setdefault(child, set()).add(parent)

    def add_instance(self, instance: str, cls: str) -> None:
        self.instance_of.setdefault(instance, set()).add(cls)

    def all_superclasses(self, cls: str) -> Set[str]:
        seen: Set[str] = set()
        agenda = list(self.subclass_of.get(cls, set()))
        while agenda:
            parent = agenda.pop()
            if parent not in seen:
                seen.add(parent)
                agenda.extend(self.subclass_of.get(parent, set()))
        return seen

    def infer_instance_types(self, instance: str) -> Set[str]:
        direct = self.instance_of.get(instance, set())
        inferred = set(direct)
        for cls in direct:
            inferred.update(self.all_superclasses(cls))
        return inferred


class SimpleKnowledgeGraph:
    """Small semantic graph represented as subject-predicate-object triples."""

    def __init__(self) -> None:
        self.triples: List[Triple] = []

    def add(self, subject: str, predicate: str, obj: str) -> None:
        self.triples.append(Triple(subject, predicate, obj))

    def query(self, subject=None, predicate=None, obj=None) -> List[Triple]:
        return [
            t for t in self.triples
            if (subject is None or t.subject == subject)
            and (predicate is None or t.predicate == predicate)
            and (obj is None or t.object == obj)
        ]

    def neighbours(self, entity: str) -> List[Tuple[str, str]]:
        return [(t.predicate, t.object) for t in self.triples if t.subject == entity]


def semantic_web_triples() -> List[Triple]:
    return [
        Triple("AI_System", "subClassOf", "Software_System"),
        Triple("Decision_Tree", "subClassOf", "Machine_Learning_Method"),
        Triple("Random_Forest", "uses", "Decision_Tree"),
        Triple("Knowledge_Graph", "represents", "Entities_and_Relationships"),
    ]


def demo() -> dict:
    ontology = SimpleOntology()
    ontology.add_subclass("DecisionTree", "MachineLearningMethod")
    ontology.add_subclass("MachineLearningMethod", "AIComputationalApproach")
    ontology.add_instance("iris_tree_model", "DecisionTree")

    kg = SimpleKnowledgeGraph()
    kg.add("RandomForest", "uses", "DecisionTree")
    kg.add("DecisionTree", "supports", "Classification")
    kg.add("Ontology", "supports", "KnowledgeRepresentation")

    return {
        "inferred_types": sorted(ontology.infer_instance_types("iris_tree_model")),
        "random_forest_neighbours": kg.neighbours("RandomForest"),
        "semantic_triples": semantic_web_triples(),
    }


if __name__ == "__main__":
    print(demo())
