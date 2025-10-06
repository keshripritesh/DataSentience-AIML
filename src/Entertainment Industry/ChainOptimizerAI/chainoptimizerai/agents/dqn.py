from __future__ import annotations

# Placeholder DQN stub to keep tests runnable without heavy deps
class DQNAgent:
    def __init__(self):
        pass

    def train_episode(self, env, max_steps=None):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # naive random action similar to random policy
            prod = {f.id: 0.0 for f in env.factories}
            ship_fw = {(e.source_id, e.target_id): 0.0 for e in env.edges_fw}
            ship_wr = {(e.source_id, e.target_id): 0.0 for e in env.edges_wr}
            state, reward, done, _ = env.step(prod, ship_fw, ship_wr)
            total_reward += reward
        return total_reward


