"""Logic and reasoning examples for AI systems."""
from collections import Counter
from typing import Sequence


def modus_ponens(if_p_then_q: tuple[str, str], observed_p: str) -> str | None:
    """If P implies Q, and P is observed, infer Q."""
    p, q = if_p_then_q
    return q if observed_p == p else None


def modus_tollens(if_p_then_q: tuple[str, str], observed_not_q: str) -> str | None:
    """If P implies Q, and not Q is observed, infer not P."""
    p, q = if_p_then_q
    return f"not {p}" if observed_not_q == f"not {q}" else None


def transitive_inference(a_to_b: tuple[str, str], b_to_c: tuple[str, str]) -> tuple[str, str] | None:
    """If A implies B and B implies C, infer A implies C."""
    a, b1 = a_to_b
    b2, c = b_to_c
    return (a, c) if b1 == b2 else None


def inductive_majority_rule(examples: Sequence[str]) -> tuple[str, float]:
    """Generalise from examples using the most common observed class."""
    if not examples:
        raise ValueError("examples cannot be empty")
    counts = Counter(examples)
    label, n = counts.most_common(1)[0]
    return label, n / len(examples)


def test_hypothesis(observations: Sequence[float], predicted_value: float, tolerance: float) -> dict:
    """Check whether observations support a numerical hypothesis within a tolerance."""
    matches = [abs(x - predicted_value) <= tolerance for x in observations]
    return {"matches": sum(matches), "total": len(matches), "supported": all(matches)}


def bayes_rule(prior: float, likelihood: float, evidence: float) -> float:
    """Posterior = likelihood * prior / evidence."""
    if evidence <= 0:
        raise ValueError("evidence must be positive")
    return max(0.0, min(1.0, likelihood * prior / evidence))


def demo() -> dict:
    return {
        "modus_ponens": modus_ponens(("is_human", "is_mortal"), "is_human"),
        "modus_tollens": modus_tollens(("rains", "grass_wet"), "not grass_wet"),
        "transitive": transitive_inference(("A", "B"), ("B", "C")),
        "induction": inductive_majority_rule(["white_swan", "white_swan", "black_swan"]),
        "hypothesis": test_hypothesis([9.7, 9.8, 9.9], 9.8, 0.2),
        "bayes": round(bayes_rule(0.01, 0.95, 0.05), 3),
    }


if __name__ == "__main__":
    print(demo())
