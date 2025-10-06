from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np


def create_bins(low: float, high: float, num_bins: int) -> np.ndarray:
    return np.linspace(low, high, num_bins + 1)


def to_bin(value: float, bins: np.ndarray) -> int:
    idx = int(np.digitize([value], bins)[0]) - 1
    return max(0, min(len(bins) - 2, idx))


@dataclass
class QLearningConfig:
    angle_bins: int = 19
    force_bins: int = 19
    learning_rate: float = 0.2
    discount: float = 0.95
    epsilon: float = 0.2


class TabularQAgent:
    def __init__(self, angle_range: Tuple[float, float], force_range: Tuple[float, float], config: QLearningConfig | None = None) -> None:
        self.config = config or QLearningConfig()
        self.angle_bins = create_bins(angle_range[0], angle_range[1], self.config.angle_bins)
        self.force_bins = create_bins(force_range[0], force_range[1], self.config.force_bins)
        self.q_table = np.zeros((self.config.angle_bins, self.config.force_bins), dtype=np.float32)

    def select_action(self, epsilon: float | None = None) -> Tuple[float, float]:
        eps = self.config.epsilon if epsilon is None else epsilon
        if np.random.rand() < eps:
            angle_idx = np.random.randint(0, self.q_table.shape[0])
            force_idx = np.random.randint(0, self.q_table.shape[1])
        else:
            angle_idx, force_idx = np.unravel_index(np.argmax(self.q_table), self.q_table.shape)
        angle = float(self._from_bin(angle_idx, self.angle_bins))
        force = float(self._from_bin(force_idx, self.force_bins))
        return angle, force

    def update(self, action: Tuple[float, float], reward: float) -> None:
        a_idx = to_bin(action[0], self.angle_bins)
        f_idx = to_bin(action[1], self.force_bins)
        current = self.q_table[a_idx, f_idx]
        best_next = float(np.max(self.q_table))
        target = reward + self.config.discount * best_next
        self.q_table[a_idx, f_idx] = (1 - self.config.learning_rate) * current + self.config.learning_rate * target

    @staticmethod
    def _from_bin(index: int, bins: np.ndarray) -> float:
        left = bins[index]
        right = bins[index + 1]
        return (left + right) * 0.5


