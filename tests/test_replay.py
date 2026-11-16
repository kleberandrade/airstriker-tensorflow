"""Replay and target-network behavior on synthetic arrays, not emulator data."""

import numpy as np
import pytest

from airstriker_tensorflow.agent import Agent, DQNConfig
from airstriker_tensorflow.replay import ReplayBuffer


def test_replay_is_bounded_copies_observations_and_samples_without_replacement():
    replay = ReplayBuffer(capacity=2, seed=7)
    state = np.zeros((32, 32, 4), dtype=np.float32)
    for action in range(3):
        replay.append(state, action, float(action), state, action == 2)
    state[:] = 1
    batch = replay.sample(2)
    assert len(replay) == 2
    assert sorted(batch.actions.tolist()) == [1, 2]
    assert batch.states.dtype == np.float32
    assert np.all(batch.states == 0) and np.all(batch.next_states == 0)
    assert batch.terminated[batch.actions == 2].all()
    assert replay.storage_bytes == 2 * 2 * state.size  # compact uint8 pairs
    with pytest.raises(ValueError):
        replay.sample(3)


def test_seeded_sampling_repeats_and_preserves_normalized_pixel_values():
    left, right = ReplayBuffer(5, seed=13), ReplayBuffer(5, seed=13)
    state = np.full((32, 32, 4), 127 / 255, dtype=np.float32)
    for action in range(5):
        left.append(state, action, action + 0.5, state, False)
        right.append(state, action, action + 0.5, state, False)
    a, b = left.sample(3), right.sample(3)
    np.testing.assert_array_equal(a.actions, b.actions)
    np.testing.assert_allclose(a.states, state[None] + np.zeros((3, 32, 32, 4)))
    np.testing.assert_allclose(a.rewards, a.actions + 0.5)


def test_warmup_defers_updates_and_target_sync_uses_optimizer_update_count():
    agent = Agent(
        3,
        seed=7,
        config=DQNConfig(
            replay_capacity=4, batch_size=2, learning_starts=3, target_sync_interval=2
        ),
    )
    state = np.ones((32, 32, 4), dtype=np.float32)
    initial = [w.copy() for w in agent.model.get_weights()]
    for _ in range(2):
        assert agent.learn(state, 1, 5.0, state, True) is None
    assert agent.updates == 0
    for old, new in zip(initial, agent.model.get_weights()):
        np.testing.assert_array_equal(old, new)
    assert np.isfinite(agent.learn(state, 1, 5.0, state, True))
    assert agent.updates == 1 and agent.target_syncs == 0
    for old, target in zip(initial, agent.target_model.get_weights()):
        np.testing.assert_array_equal(old, target)
    agent.learn(state, 1, 5.0, state, True)
    assert agent.updates == 2 and agent.target_syncs == 1
    for online, target in zip(agent.model.get_weights(), agent.target_model.get_weights()):
        np.testing.assert_array_equal(online, target)
    assert any(not np.array_equal(old, new) for old, new in zip(initial, agent.model.get_weights()))


def test_bootstrap_comes_from_frozen_target_predictions_with_real_keras_loss():
    agent = Agent(
        3,
        seed=7,
        config=DQNConfig(
            replay_capacity=2, batch_size=1, learning_starts=1, target_sync_interval=100
        ),
    )
    online = [np.zeros_like(w) for w in agent.model.get_weights()]
    target = [np.zeros_like(w) for w in agent.target_model.get_weights()]
    online[-1][:] = [1.0, 2.0, 3.0]
    target[-1][:] = [10.0, 20.0, 30.0]
    agent.model.set_weights(online)
    agent.target_model.set_weights(target)
    state = np.zeros((32, 32, 4), dtype=np.float32)
    loss = agent.learn(state, 1, 0.0, state, False)
    # Target for action 1 is 0 + .99 * 30 = 29.7, not .99 * 3.
    assert loss == pytest.approx((27.7 - 0.5) / 3, rel=1e-5)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"replay_capacity": 0},
        {"learning_starts": 0},
        {"target_sync_interval": 0},
        {"batch_size": 64, "replay_capacity": 32},
        {"batch_size": 32, "learning_starts": 16},
        {"replay_capacity": 128, "learning_starts": 256},
    ],
)
def test_invalid_replay_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        DQNConfig(**options)


def test_replay_rejects_empty_zero_and_negative_sampling():
    with pytest.raises(ValueError, match="positive"):
        ReplayBuffer(0)
    buffer = ReplayBuffer(2)
    for size in (0, -1, 1):
        with pytest.raises(ValueError, match="sample size"):
            buffer.sample(size)
