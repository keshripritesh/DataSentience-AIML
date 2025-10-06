import argparse
import numpy as np

from arrowflightai.env.arrow_env import ArrowFlightEnv
from arrowflightai.agents.q_learning import TabularQAgent, QLearningConfig


def run_q_learning(episodes: int = 200) -> None:
    env = ArrowFlightEnv()
    agent = TabularQAgent(env.angle_range, env.force_range, QLearningConfig())
    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        action = agent.select_action()
        next_obs, reward, done, info = env.step(action)
        agent.update(action, reward)
        rewards.append(reward)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}: reward={np.mean(rewards[-50:]):.3f} last_dist={info['distance']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--agent", type=str, default="q_learning", choices=["q_learning"])  # DQN stubbed for demo
    args = parser.parse_args()
    if args.agent == "q_learning":
        run_q_learning(args.episodes)


