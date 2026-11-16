"""Convert emulator RGB frames to a small temporal observation."""

from collections import deque
import numpy as np
from PIL import Image


class FrameStack:
    def __init__(self):
        self.frames = deque(maxlen=4)

    def _frame(self, image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB observation")
        return (
            np.asarray(Image.fromarray(image).convert("L").resize((32, 32)), dtype=np.float32)
            / 255.0
        )

    def reset(self, image):
        frame = self._frame(image)
        self.frames.clear()
        self.frames.extend([frame] * 4)
        return np.stack(self.frames, axis=-1)

    def append(self, image):
        self.frames.append(self._frame(image))
        return np.stack(self.frames, axis=-1)
