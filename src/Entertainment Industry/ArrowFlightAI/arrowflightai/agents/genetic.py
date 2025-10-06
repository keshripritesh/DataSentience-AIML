from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class GAConfig:
    population_size: int = 40
    elite_fraction: float = 0.2
    mutation_std: float = 2.0
    generations: int = 30


class GeneticOptimizer:
    def __init__(self, angle_range: Tuple[float, float], force_range: Tuple[float, float], config: GAConfig | None = None) -> None:
        self.config = config or GAConfig()
        self.angle_range = angle_range
        self.force_range = force_range

    def optimize(self, evaluate: Callable[[Tuple[float, float]], float], rng: np.random.RandomState | None = None) -> Tuple[float, float, float]:
        rng = rng or np.random.RandomState(0)
        pop = self._init_population(rng)

        best_score = -np.inf
        best_action = (0.0, 0.0)
        for _ in range(self.config.generations):
            scores = np.array([evaluate(tuple(ind)) for ind in pop])
            elite_count = max(1, int(self.config.elite_fraction * len(pop)))
            elite_idx = np.argsort(scores)[-elite_count:]
            elites = pop[elite_idx]

            if float(np.max(scores)) > best_score:
                best_score = float(np.max(scores))
                best_action = tuple(pop[int(np.argmax(scores))])  # type: ignore[assignment]

            children = self._crossover_and_mutate(elites, len(pop) - elite_count, rng)
            pop = np.vstack([elites, children])

        return best_action[0], best_action[1], float(best_score)

    def _init_population(self, rng: np.random.RandomState) -> np.ndarray:
        angles = rng.uniform(self.angle_range[0], self.angle_range[1], size=self.config.population_size)
        forces = rng.uniform(self.force_range[0], self.force_range[1], size=self.config.population_size)
        return np.stack([angles, forces], axis=1)

    def _crossover_and_mutate(self, elites: np.ndarray, num_children: int, rng: np.random.RandomState) -> np.ndarray:
        children: List[np.ndarray] = []
        while len(children) < num_children:
            p1, p2 = elites[rng.randint(0, len(elites))], elites[rng.randint(0, len(elites))]
            alpha = rng.rand()
            child = alpha * p1 + (1 - alpha) * p2
            child[0] = np.clip(child[0] + rng.randn() * self.config.mutation_std, self.angle_range[0], self.angle_range[1])
            child[1] = np.clip(child[1] + rng.randn() * self.config.mutation_std, self.force_range[0], self.force_range[1])
            children.append(child)
        return np.stack(children, axis=0)


