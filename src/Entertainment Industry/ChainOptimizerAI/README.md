# ChainOptimizerAI Pro: Adaptive Supply Chain RL Simulator

An advanced, research-grade simulator for multi-stage supply chain optimization with reinforcement learning, baseline heuristics, and lightweight visualization. Design goals: clarity, extensibility, and reproducibility.

## Features

- Turn-based simulator with factories, warehouses, retailers
- Costs: production, transport, holding, unmet demand penalties
- Agents: Tabular Q-learning, DQN stub, greedy LP-like heuristic
- Visualization: network graph, cost curves
- Fully synthetic configs; no external data required

## Install

```bash
pip install -r requirements.txt
```

## Quickstart

Run a random-agent simulation:

```bash
python -m chainoptimizerai.env.supply_chain_env --episodes 3 --config chainoptimizerai/examples/small_chain.json
```

Train Q-learning agent:

```bash
python -m chainoptimizerai.experiments.train_q_agent --episodes 200
```

Visualize network:

```bash
python -m chainoptimizerai.visualization.plot_network --config chainoptimizerai/examples/small_chain.json
```

## Tests

```bash
pytest -q
```

## Roadmap

- True DQN (PyTorch) with replay buffer and target nets
- Multi-agent coordination and disruptions
- Benchmarking suite and LP/MIP baselines


