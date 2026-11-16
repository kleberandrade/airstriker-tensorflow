"""Small online Q-learning baseline, without replay or a target network."""

from pathlib import Path

import numpy as np
import tensorflow as tf


def bootstrap_targets(rewards, terminated, next_values, gamma):
    return np.asarray(rewards) + gamma * (1 - np.asarray(terminated, dtype=np.float32)) * np.max(
        next_values, axis=-1
    )


class Agent:
    def __init__(self, action_count, seed=0, checkpoint=None):
        tf.keras.utils.set_random_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.action_count = action_count
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

    def values(self, state):
        return self.model(state[None], training=False).numpy()[0]

    def act(self, state, epsilon=0.1):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        return int(np.argmax(self.values(state)))

    def learn(self, state, action, reward, next_state, terminated):
        target = self.values(state)[None]
        target[0, action] = bootstrap_targets(
            [reward], [terminated], self.values(next_state)[None], 0.99
        )[0]
        return float(self.model.train_on_batch(state[None], target))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
