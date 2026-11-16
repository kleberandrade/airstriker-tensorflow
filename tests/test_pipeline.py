import numpy as np
import pytest

from airstriker_tensorflow.agent import Agent, DQNConfig, bootstrap_targets
from airstriker_tensorflow.environments import SyntheticEnvironment
from airstriker_tensorflow.preprocessing import FrameStack
from airstriker_tensorflow.training import run


def test_frames_are_normalized_stacked_and_reset():
    stack = FrameStack()
    black = np.zeros((40, 50, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    state = stack.reset(black)
    assert state.shape == (32, 32, 4)
    assert state.dtype == np.float32
    assert np.all(state == 0)
    state = stack.append(white)
    assert np.all(state[..., :3] == 0)
    assert np.all(state[..., 3] == 1)
    assert np.all(stack.reset(white) == 1)


def test_bad_observation_is_rejected():
    with pytest.raises(ValueError, match="RGB"):
        FrameStack().reset(np.zeros((10, 10)))


def test_synthetic_environment_is_seeded_and_bounded():
    env = SyntheticEnvironment()
    first, info = env.reset(seed=42)
    assert info["synthetic"] is True
    assert np.array_equal(first, env.reset(seed=42)[0])
    for _ in range(7):
        _, reward, terminated, truncated, _ = env.step(1)
        assert reward == 1.0
        assert not terminated and not truncated
    assert env.step(1)[3] is True
    with pytest.raises(ValueError):
        env.step(99)
    env.close()


def test_terminal_bootstrap_and_time_limit():
    # Time-limit truncations bootstrap; true terminals do not.
    actual = bootstrap_targets([1.0, 2.0], [True, False], [[10.0, 20.0], [4.0, 8.0]], 0.5)
    np.testing.assert_allclose(actual, [1.0, 6.0])


def test_training_updates_weights_and_checkpoint_roundtrip(tmp_path):
    agent = Agent(3, seed=7, config=DQNConfig(batch_size=1, learning_starts=1))
    state = np.ones((32, 32, 4), dtype=np.float32)
    before = agent.values(state).copy()
    loss = agent.learn(state, 1, 5.0, state, True)
    assert np.isfinite(loss)
    assert not np.array_equal(before, agent.values(state))
    checkpoint = tmp_path / "agent.keras"
    agent.save(checkpoint)
    loaded = Agent(3, seed=9, checkpoint=checkpoint)
    np.testing.assert_allclose(agent.values(state), loaded.values(state))
    assert loaded.act(state, epsilon=0) == int(np.argmax(loaded.values(state)))
    with pytest.raises(ValueError, match="action"):
        Agent(2, checkpoint=checkpoint)


def test_short_train_and_evaluation_have_explicit_limits():
    agent = Agent(3, seed=1, config=DQNConfig(batch_size=2, learning_starts=2))
    env = SyntheticEnvironment()
    result = run(env, agent, steps=10, seed=3, training=True)
    assert result["steps"] == 10
    assert result["episodes"] == 1
    assert np.isfinite(result["loss"])
    before = [w.copy() for w in agent.model.get_weights()]
    result = run(env, agent, steps=2, seed=3, training=False)
    assert result["steps"] == 2
    for old, new in zip(before, agent.model.get_weights()):
        np.testing.assert_array_equal(old, new)
    with pytest.raises(ValueError, match="positive"):
        run(env, agent, steps=0)


def test_replay_warmup_records_steps_without_claiming_optimizer_updates():
    agent = Agent(3, seed=7)
    result = run(SyntheticEnvironment(), agent, steps=8, seed=3, training=True)
    assert result["steps"] == 8 and result["loss"] is None
    assert agent.updates == 0 and len(agent.replay) == 8
