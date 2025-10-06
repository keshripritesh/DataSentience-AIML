import math
import numpy as np

from arrowflightai.env.arrow_env import ArrowFlightEnv


def test_reset_returns_obs():
    env = ArrowFlightEnv()
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3,)


def test_step_returns_reward_and_done():
    env = ArrowFlightEnv()
    env.reset()
    obs, reward, done, info = env.step((45.0, 50.0))
    assert isinstance(reward, float)
    assert done is True
    assert "trajectory" in info and info["trajectory"].ndim == 2


def test_wind_affects_landing():
    env = ArrowFlightEnv()
    env.reset()
    # Fix target to avoid reward confounders
    assert env.state is not None
    env.state.target_x = 60.0
    env.state.target_y = 0.0

    env.state.wind = -2.0
    _, _, _, info1 = env.step((45.0, 40.0))
    env.reset()
    assert env.state is not None
    env.state.target_x = 60.0
    env.state.target_y = 0.0
    env.state.wind = 2.0
    _, _, _, info2 = env.step((45.0, 40.0))

    assert info1["landing"][0] != info2["landing"][0]


def test_reward_improves_when_closer():
    env = ArrowFlightEnv()
    env.reset()
    assert env.state is not None
    env.state.target_x = 50.0
    env.state.target_y = 0.0

    # Shot A
    _, r1, _, info1 = env.step((30.0, 20.0))
    # Shot B closer to 50m on flat ground: 45 deg approx maximizes range
    env.reset()
    assert env.state is not None
    env.state.target_x = 50.0
    env.state.target_y = 0.0
    _, r2, _, info2 = env.step((45.0, 25.0))

    assert r2 > r1


