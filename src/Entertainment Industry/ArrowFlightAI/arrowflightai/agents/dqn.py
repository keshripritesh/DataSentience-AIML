from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    hidden_size: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class SimpleDQNAgent:
    """A simple DQN that maps observation -> discretized action index."""

    def __init__(self, obs_dim: int, angle_bins: int, force_bins: int, config: DQNConfig | None = None) -> None:
        self.config = config or DQNConfig()
        self.angle_bins = angle_bins
        self.force_bins = force_bins
        self.num_actions = angle_bins * force_bins
        self.model = MLP(obs_dim, self.num_actions, self.config.hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, obs: np.ndarray, epsilon: float = 0.1) -> Tuple[int, int]:
        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.model(torch.from_numpy(obs).float().unsqueeze(0))
                action_index = int(torch.argmax(q_values, dim=1).item())
        return divmod(action_index, self.force_bins)

    def train_step(self, obs: np.ndarray, action_idx: int, reward: float, next_obs: np.ndarray, done: bool) -> float:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        next_obs_t = torch.from_numpy(next_obs).float().unsqueeze(0)

        q_values = self.model(obs_t)
        with torch.no_grad():
            next_q = self.model(next_obs_t)
            target = reward + (0.0 if done else self.config.gamma * float(torch.max(next_q)))

        target_q = q_values.clone().detach()
        target_q[0, action_idx] = target

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


