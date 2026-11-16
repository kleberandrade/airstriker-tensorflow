"""Explicit synthetic fixture and optional real emulator integration."""

import numpy as np


class SyntheticEnvironment:
    """Test fixture only: three actions, random RGB frames, eight-step episodes."""

    action_count = 3

    def reset(self, *, seed=None):
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        return self._frame(), {"synthetic": True}

    def _frame(self):
        return self.rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    def step(self, action):
        if action not in range(self.action_count):
            raise ValueError("Invalid action")
        self.steps += 1
        return self._frame(), float(action == 1), False, self.steps >= 8, {"synthetic": True}

    def close(self):
        pass


def make_retro(game, state=None):
    import stable_retro

    options = {} if state is None else {"state": state}
    env = stable_retro.make(
        game=game,
        render_mode="rgb_array",
        use_restricted_actions=stable_retro.Actions.DISCRETE,
        **options,
    )
    env.action_count = int(env.action_space.n)
    return env
