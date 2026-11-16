"""Offline pipeline check. This synthetic fixture does not emulate Mario."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mario_tensorflow.cli import main

with TemporaryDirectory() as directory:
    checkpoint = str(Path(directory) / "smoke.keras")
    main(["train", "--env", "synthetic", "--steps", "8", "--checkpoint", checkpoint])
    main(["evaluate", "--env", "synthetic", "--steps", "8", "--checkpoint", checkpoint])
