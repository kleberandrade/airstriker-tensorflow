"""Seeded, bounded replay with compact copies of normalized pixel observations."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.transitions = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.transitions)

    @property
    def storage_bytes(self):
        return sum(
            state.nbytes + next_state.nbytes for state, _, _, next_state, _ in self.transitions
        )

    def append(self, state, action, reward, next_state, terminated):
        self.transitions.append(
            (_pixels(state), int(action), float(reward), _pixels(next_state), bool(terminated))
        )

    def sample(self, size):
        if not 0 < size <= len(self):
            raise ValueError("sample size must be positive and no larger than replay")
        indices = self.rng.choice(len(self), size=size, replace=False)
        rows = [self.transitions[int(index)] for index in indices]
        states, actions, rewards, next_states, terminated = zip(*rows, strict=True)
        return ReplayBatch(
            np.stack(states).astype(np.float32) / 255.0,
            np.asarray(actions, dtype=np.int32),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32) / 255.0,
            np.asarray(terminated, dtype=bool),
        )


def _pixels(state):
    return np.rint(np.asarray(state) * 255).astype(np.uint8)
