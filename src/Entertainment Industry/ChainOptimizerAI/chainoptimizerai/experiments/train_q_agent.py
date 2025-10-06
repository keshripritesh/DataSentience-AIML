import argparse
import json
from chainoptimizerai.env.supply_chain_env import build_env_from_config
from chainoptimizerai.agents.q_learning import TabularQAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='chainoptimizerai/examples/small_chain.json')
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    env = build_env_from_config(cfg)

    agent = TabularQAgent()
    rewards = []
    for ep in range(args.episodes):
        r = agent.train_episode(env)
        rewards.append(r)
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}: reward={r:.2f}")


if __name__ == '__main__':
    main()


