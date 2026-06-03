# Metaheuristics

Metaheuristics are optimisation methods that search for sufficiently good solutions when exact optimisation is difficult, expensive, non-differentiable, or poorly structured.

![Genetic algorithm cycle](images/genetic-algorithm-cycle.svg)

## Methods covered

| Method | Key idea |
|---|---|
| Genetic algorithm | Evolve a population using selection, crossover, and mutation |
| Evolution strategy | Sample candidate solutions around a moving distribution |
| Simulated annealing | Accept some worse moves early to escape local optima |
| Particle swarm optimisation | Particles move using personal and global best positions |

## When useful

- Non-convex optimisation
- Non-differentiable objective functions
- Engineering search problems
- Hyperparameter search
- Scheduling and routing
- Design-space exploration

## Related code

```bash
PYTHONPATH=src python examples/metaheuristics_demo.py
```

Main module:

```text
src/ai_computational_approaches/metaheuristics.py
```
