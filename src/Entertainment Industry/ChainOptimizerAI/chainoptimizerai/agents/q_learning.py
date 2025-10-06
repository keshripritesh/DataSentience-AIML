from __future__ import annotations

from typing import Dict, Tuple, DefaultDict
from collections import defaultdict
import numpy as np


class TabularQAgent:
    def __init__(self, learning_rate: float = 0.1, discount: float = 0.95, epsilon: float = 0.1, seed: int = 0):
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.random = np.random.default_rng(seed)
        self.q: DefaultDict[Tuple, DefaultDict[Tuple, float]] = defaultdict(lambda: defaultdict(float))

    def discretize_state(self, state) -> Tuple:
        # Simple aggregation: bin inventories to ints
        wh = tuple(int(round(v)) for v in sorted(state.inventory_warehouse.values()))
        rt = tuple(int(round(v)) for v in sorted(state.inventory_retailer.values()))
        d = tuple(int(round(v)) for v in sorted(state.demand_this_step.values()))
        return (wh, rt, d)

    def sample_action(self, env) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
        prod = {f.id: float(self.random.integers(0, max(1, int(f.production_capacity + 1)))) for f in env.factories}
        ship_fw = {(e.source_id, e.target_id): float(self.random.integers(0, 3)) for e in env.edges_fw}
        ship_wr = {(e.source_id, e.target_id): float(self.random.integers(0, 3)) for e in env.edges_wr}
        return prod, ship_fw, ship_wr

    def best_action(self, state_key):
        actions_q = self.q[state_key]
        if not actions_q:
            return None, 0.0
        best_a = max(actions_q.items(), key=lambda kv: kv[1])[0]
        return best_a, actions_q[best_a]

    def train_episode(self, env, max_steps: int | None = None) -> float:
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done and (max_steps is None or steps < max_steps):
            s_key = self.discretize_state(state)
            if self.random.random() < self.epsilon or not self.q[s_key]:
                action = self.sample_action(env)
            else:
                action, _ = self.best_action(s_key)
                if action is None:
                    action = self.sample_action(env)

            next_state, reward, done, _ = env.step(*action)
            total_reward += reward
            ns_key = self.discretize_state(next_state)

            # Q update
            _, max_next_q = self.best_action(ns_key)
            current_q = self.q[s_key][action]
            target = reward + self.gamma * max_next_q * (0.0 if done else 1.0)
            self.q[s_key][action] = current_q + self.alpha * (target - current_q)

            state = next_state
            steps += 1
        return total_reward


