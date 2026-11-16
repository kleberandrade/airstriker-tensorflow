"""Adapter contract tests only: no emulator or ROM is involved here."""

from types import SimpleNamespace

import numpy as np

from airstriker_tensorflow.agent import Agent
from airstriker_tensorflow.environments import make_retro
from airstriker_tensorflow.training import run


def test_retro_factory_selects_discrete_rgb_actions(monkeypatch):
    import stable_retro

    calls = []

    def make(**options):
        calls.append(options)
        return SimpleNamespace(action_space=SimpleNamespace(n=7))

    monkeypatch.setattr(stable_retro, "make", make)
    assert make_retro("user-game", "user-state").action_count == 7
    assert calls == [
        {
            "game": "user-game",
            "state": "user-state",
            "render_mode": "rgb_array",
            "use_restricted_actions": stable_retro.Actions.DISCRETE,
        }
    ]
    make_retro("default-state")
    assert "state" not in calls[1]


def test_true_terminal_resets_without_cross_episode_frame_history():
    states = []
    seeds = []
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full_like(black, 255)

    class Environment:
        def reset(self, *, seed):
            seeds.append(seed)
            return black, {}

        def step(self, action):
            return white, 2.0, True, False, {}

    class RecordingAgent:
        def act(self, state, epsilon):
            states.append(state)
            return 0

        def learn(self, state, action, reward, next_state, terminated):
            assert terminated
            assert np.all(next_state[..., 3] == 1)
            assert np.all(next_state[..., :3] == 0)
            return 1.0

    result = run(Environment(), RecordingAgent(), steps=2, seed=5, training=True)
    assert result == {"steps": 2, "episodes": 2, "reward": 4.0, "loss": 1.0}
    assert seeds == [5, 6, 7]
    assert all(np.all(state == 0) for state in states)


def test_seed_reproduces_initial_values_and_random_actions():
    state = np.ones((32, 32, 4), dtype=np.float32)
    first = Agent(3, seed=12)
    second = Agent(3, seed=12)
    np.testing.assert_array_equal(first.values(state), second.values(state))
    assert [first.act(state, 1.0) for _ in range(12)] == [second.act(state, 1.0) for _ in range(12)]
