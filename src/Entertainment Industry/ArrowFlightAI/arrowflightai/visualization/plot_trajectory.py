import argparse
import matplotlib.pyplot as plt
from typing import Optional

from arrowflightai.env.arrow_env import ArrowFlightEnv


def plot(angle: float, force: float, target_x: float, target_y: float, wind: Optional[float] = None) -> None:
    env = ArrowFlightEnv()
    env.state = env.state  # no-op for type checkers
    env.reset()
    if wind is not None and env.state is not None:
        env.state.wind = wind
    if env.state is not None:
        env.state.target_x = target_x
        env.state.target_y = target_y
    _, reward, _, info = env.step((angle, force))
    traj = info["trajectory"]

    plt.figure(figsize=(8, 4))
    plt.plot(traj[:, 0], traj[:, 1], label="trajectory")
    plt.scatter([target_x], [target_y], c="red", label="target")
    plt.scatter([info["landing"][0]], [info["landing"][1]], c="green", label="landing")
    plt.title(f"Angle={angle:.1f} Force={force:.1f} Reward={reward:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--force", type=float, required=True)
    parser.add_argument("--target_x", type=float, required=True)
    parser.add_argument("--target_y", type=float, required=True)
    parser.add_argument("--wind", type=float, default=None)
    args = parser.parse_args()
    plot(args.angle, args.force, args.target_x, args.target_y, args.wind)


