"""Small DQN with uniform experience replay and a periodically copied target network."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from .replay import ReplayBuffer

ALGORITHM_NAME = "dqn-replay-target"


@dataclass(frozen=True)
class DQNConfig:
    replay_capacity: int = 5000
    batch_size: int = 32
    learning_starts: int = 256
    target_sync_interval: int = 250

    def __post_init__(self):
        if (
            min(
                self.replay_capacity,
                self.batch_size,
                self.learning_starts,
                self.target_sync_interval,
            )
            <= 0
        ):
            raise ValueError("DQN parameters must be positive")
        if not self.batch_size <= self.learning_starts <= self.replay_capacity:
            raise ValueError("Require batch_size <= learning_starts <= replay_capacity")


def bootstrap_targets(rewards, terminated, next_values, gamma):
    return np.asarray(rewards) + gamma * (1 - np.asarray(terminated, dtype=np.float32)) * np.max(
        next_values, axis=-1
    )


class Agent:
    def __init__(self, action_count, seed=0, checkpoint=None, config=None):
        tf.keras.utils.set_random_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.action_count = action_count
        self.config = config or DQNConfig()
        self.replay = ReplayBuffer(self.config.replay_capacity, seed=seed)
        self.updates = 0
        self.target_syncs = 0
        if checkpoint:
            self.model = tf.keras.models.load_model(checkpoint)
            if self.model.output_shape[-1] != action_count:
                raise ValueError("Checkpoint action count differs from environment")
        else:
            self.model = tf.keras.Sequential(
                [
                    tf.keras.Input((32, 32, 4)),
                    tf.keras.layers.Conv2D(8, 5, strides=4, activation="relu"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dense(action_count),
                ]
            )
            self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber")
        self.target_model = tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.trainable = False

    def values(self, state):
        return self.model(state[None], training=False).numpy()[0]

    def act(self, state, epsilon=0.1):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        return int(np.argmax(self.values(state)))

    def learn(self, state, action, reward, next_state, terminated):
        self.replay.append(state, action, reward, next_state, terminated)
        if len(self.replay) < self.config.learning_starts:
            return None
        batch = self.replay.sample(self.config.batch_size)
        targets = self.model(batch.states, training=False).numpy()
        next_values = self.target_model(batch.next_states, training=False).numpy()
        targets[np.arange(self.config.batch_size), batch.actions] = bootstrap_targets(
            batch.rewards, batch.terminated, next_values, 0.99
        )
        self.model.reset_metrics()
        loss = float(self.model.train_on_batch(batch.states, targets))
        self.updates += 1
        self._sync_target()
        return loss

    def _sync_target(self):
        if self.updates % self.config.target_sync_interval == 0:
            self.target_model.set_weights(self.model.get_weights())
            self.target_syncs += 1

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
