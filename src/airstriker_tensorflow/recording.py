"""Record real observations and transitions without changing environment behavior."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocessing import FrameStack


class RecordedEnvironment:
    def __init__(self, env, directory, capture_every=128):
        if capture_every <= 0:
            raise ValueError("capture_every must be positive")
        self.env = env
        self.action_count = env.action_count
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = (self.directory / "transitions.jsonl").open("x")
        self.capture_every = capture_every
        self.steps = 0
        self.resets = 0
        self.captures = []

    def reset(self, *, seed=None):
        result = self.env.reset(seed=seed)
        self.resets += 1
        self._capture(result[0])
        self._write({"event": "reset", "step": self.steps, "seed": seed, "info": result[1]})
        return result

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        image, reward, terminated, truncated, info = result
        self._write(
            {
                "event": "step",
                "step": self.steps,
                "action": int(action),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )
        if self.steps == 1 or self.steps % self.capture_every == 0:
            self._capture(image)
        return result

    def _write(self, record):
        self.trace.write(json.dumps(record, default=_numpy_scalar) + "\n")

    def _capture(self, image):
        name = f"frame-{self.steps:06d}-reset-{self.resets:03d}"
        rgb = name + ".png"
        processed = name + "-processed.png"
        Image.fromarray(image).save(self.directory / rgb)
        pixels = FrameStack().reset(image)[..., -1]
        Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).save(self.directory / processed)
        self.captures.append(
            {"step": self.steps, "rgb": rgb, "processed": processed, "shape": list(image.shape)}
        )

    def close(self):
        try:
            self.env.close()
        finally:
            self.trace.close()


def _numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
