"""Metaheuristic optimisation examples."""
import math
import random
from typing import Callable


def sphere(x: list[float]) -> float:
    """Simple minimisation benchmark with optimum at zero."""
    return sum(v * v for v in x)


def genetic_algorithm(
    objective: Callable[[list[float]], float] = sphere,
    dimensions: int = 3,
    population_size: int = 30,
    generations: int = 60,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)

    def individual():
        return [rng.uniform(-5, 5) for _ in range(dimensions)]

    def crossover(a, b):
        return [(x + y) / 2 for x, y in zip(a, b)]

    def mutate(x):
        return [v + rng.gauss(0, 0.2) for v in x]

    population = [individual() for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=objective)
        parents = population[: max(2, population_size // 3)]
        children = []
        while len(children) + len(parents) < population_size:
            a, b = rng.sample(parents, 2)
            children.append(mutate(crossover(a, b)))
        population = parents + children

    best = min(population, key=objective)
    return {"method": "genetic_algorithm", "best_solution": best, "best_score": objective(best)}


def simulated_annealing(
    objective: Callable[[list[float]], float] = sphere,
    dimensions: int = 3,
    steps: int = 300,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    current = [rng.uniform(-5, 5) for _ in range(dimensions)]
    current_score = objective(current)

    for step in range(steps):
        temp = max(0.01, 1.0 - step / steps)
        candidate = [v + rng.gauss(0, temp) for v in current]
        score = objective(candidate)
        if score < current_score or rng.random() < math.exp((current_score - score) / temp):
            current, current_score = candidate, score

    return {"method": "simulated_annealing", "best_solution": current, "best_score": current_score}


def evolution_strategy(
    objective: Callable[[list[float]], float] = sphere,
    dimensions: int = 3,
    generations: int = 60,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    mean = [rng.uniform(-2, 2) for _ in range(dimensions)]
    sigma = 1.0

    for _ in range(generations):
        candidates = [[m + rng.gauss(0, sigma) for m in mean] for _ in range(20)]
        candidates.sort(key=objective)
        mean = [sum(c[i] for c in candidates[:5]) / 5 for i in range(dimensions)]
        sigma *= 0.97

    return {"method": "evolution_strategy", "best_solution": mean, "best_score": objective(mean)}


def particle_swarm_optimisation(
    objective: Callable[[list[float]], float] = sphere,
    dimensions: int = 3,
    particles: int = 25,
    iterations: int = 60,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    positions = [[rng.uniform(-5, 5) for _ in range(dimensions)] for _ in range(particles)]
    velocities = [[0.0] * dimensions for _ in range(particles)]
    personal_best = [p[:] for p in positions]
    global_best = min(personal_best, key=objective)

    for _ in range(iterations):
        for i in range(particles):
            for d in range(dimensions):
                r1, r2 = rng.random(), rng.random()
                velocities[i][d] = (
                    0.5 * velocities[i][d]
                    + 1.2 * r1 * (personal_best[i][d] - positions[i][d])
                    + 1.2 * r2 * (global_best[d] - positions[i][d])
                )
                positions[i][d] += velocities[i][d]
            if objective(positions[i]) < objective(personal_best[i]):
                personal_best[i] = positions[i][:]
        global_best = min(personal_best, key=objective)

    return {"method": "particle_swarm_optimisation", "best_solution": global_best, "best_score": objective(global_best)}


def run_all_metaheuristics() -> list[dict]:
    return [genetic_algorithm(), evolution_strategy(), simulated_annealing(), particle_swarm_optimisation()]


if __name__ == "__main__":
    for result in run_all_metaheuristics():
        print(result)
