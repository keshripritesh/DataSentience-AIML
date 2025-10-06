# ArrowFlightAI — Neuro-Evolutionary Projectile Optimization Suite

ArrowFlightAI is a research-grade, simulation-first sandbox for precision targeting. It blends classical projectile kinematics with modern reinforcement learning and evolutionary search to learn firing strategies (angle, force) that minimize miss distance to a target. The suite includes a lightweight environment, baseline agents (Tabular Q-learning, DQN-ready scaffold, Genetic Optimizer), and visualization utilities — all with unit tests.

## Features

- Minimal, deterministic 2D projectile environment with optional wind
- Reward: negative landing distance with hit bonus
- Agents: Tabular Q-learning, simple DQN scaffold, Genetic optimizer
- Plotting utilities for trajectories and training curves
- Batteries-included tests and example experiments

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# Random rollout
python arrowflightai/env/arrow_env.py --episodes 5

# Train tabular Q-learning (demo)
python arrowflightai/experiments/train_dqn.py --episodes 200 --agent q_learning

# Visualize a trajectory
python arrowflightai/visualization/plot_trajectory.py --angle 45 --force 20 --target_x 50 --target_y 10
```

## Repo Layout

```
arrowflightai/
  env/arrow_env.py           # Simulation environment
  agents/q_learning.py       # Tabular Q-learning agent
  agents/dqn.py              # DQN scaffold
  agents/genetic.py          # Genetic optimizer
  visualization/plot_trajectory.py
  experiments/train_dqn.py   # Example training loop (Q-learning)
  tests/test_env.py          # Unit tests
```

## Roadmap

- Wind drag and turbulence models
- Moving/multi-target curricula
- Full DQN training loop + replay buffer
- 3D extension and multi-shooter arenas


