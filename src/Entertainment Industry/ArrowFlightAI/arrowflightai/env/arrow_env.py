import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import numpy as np


DEFAULT_GRAVITY = 9.81


@dataclass
class ArrowFlightState:
    target_x: float
    target_y: float
    wind: float


class ArrowFlightEnv:
    """A minimal projectile environment.

    Observation: (target_x, target_y, wind)
    Action: (angle_degrees, force)
    Reward: negative Euclidean distance from landing to target, with bonus on hit.
    """

    def __init__(
        self,
        gravity: float = DEFAULT_GRAVITY,
        force_range: Tuple[float, float] = (5.0, 100.0),
        angle_range: Tuple[float, float] = (0.0, 90.0),
        hit_threshold: float = 1.0,
        wind_range: Tuple[float, float] = (0.0, 0.0),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.gravity = gravity
        self.force_range = force_range
        self.angle_range = angle_range
        self.hit_threshold = hit_threshold
        self.wind_range = wind_range
        self._rng = rng or random.Random()
        self.state: Optional[ArrowFlightState] = None

    def reset(self, target_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((20.0, 100.0), (0.0, 50.0))) -> np.ndarray:
        tx = self._rng.uniform(*target_range[0])
        ty = self._rng.uniform(*target_range[1])
        wind = self._rng.uniform(*self.wind_range)
        self.state = ArrowFlightState(target_x=tx, target_y=ty, wind=wind)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        return np.array([self.state.target_x, self.state.target_y, self.state.wind], dtype=np.float32)

    def step(self, action: Tuple[float, float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() before step()."
        angle_deg, force = action
        angle_deg = float(np.clip(angle_deg, self.angle_range[0], self.angle_range[1]))
        force = float(np.clip(force, self.force_range[0], self.force_range[1]))

        landing_x, landing_y, trajectory = self._simulate(angle_deg, force, self.state.wind)

        dx = landing_x - self.state.target_x
        dy = landing_y - self.state.target_y
        distance = math.sqrt(dx * dx + dy * dy)
        reward = -distance
        hit = distance < self.hit_threshold
        if hit:
            reward += 10.0

        done = True  # single-step episode
        info = {
            "landing": (landing_x, landing_y),
            "distance": distance,
            "hit": hit,
            "trajectory": trajectory,
        }
        obs = self._get_obs()
        return obs, reward, done, info

    def _simulate(self, angle_degrees: float, force: float, wind: float) -> Tuple[float, float, np.ndarray]:
        angle_rad = math.radians(angle_degrees)
        vx0 = force * math.cos(angle_rad) + wind
        vy0 = force * math.sin(angle_rad)

        # Time of flight until y returns to 0 (solve vy0 * t - 0.5 * g t^2 = 0)
        if self.gravity <= 0:
            self.gravity = DEFAULT_GRAVITY
        t_flight = max(0.0, (2.0 * vy0) / self.gravity)
        t_values = np.linspace(0.0, t_flight, num=60, dtype=np.float32)
        x = vx0 * t_values
        y = vy0 * t_values - 0.5 * self.gravity * (t_values ** 2)

        # Ensure final landing y is clamped at ground level 0
        y = np.maximum(y, 0.0)

        landing_x = float(x[-1])
        landing_y = float(y[-1])
        trajectory = np.stack([x, y], axis=1)
        return landing_x, landing_y, trajectory


def random_rollout(episodes: int = 5, seed: Optional[int] = 0) -> None:
    rng = random.Random(seed)
    env = ArrowFlightEnv(rng=rng)
    for ep in range(episodes):
        obs = env.reset()
        angle = rng.uniform(*env.angle_range)
        force = rng.uniform(*env.force_range)
        _, reward, _, info = env.step((angle, force))
        print(f"Episode {ep+1}: angle={angle:.1f} force={force:.1f} reward={reward:.3f} distance={info['distance']:.3f} hit={info['hit']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run random episodes in ArrowFlightEnv")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    random_rollout(episodes=args.episodes)


