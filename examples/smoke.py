"""Offline pipeline check. This synthetic fixture does not emulate Airstriker."""

from pathlib import Path
from tempfile import TemporaryDirectory

from airstriker_tensorflow.cli import main

with TemporaryDirectory() as directory:
    checkpoint = str(Path(directory) / "smoke.keras")
    main(
        [
            "train",
            "--env",
            "synthetic",
            "--steps",
            "8",
            "--checkpoint",
            checkpoint,
            "--batch-size",
            "4",
            "--learning-starts",
            "4",
            "--replay-capacity",
            "16",
            "--target-sync-interval",
            "4",
        ]
    )
    main(["evaluate", "--env", "synthetic", "--steps", "8", "--checkpoint", checkpoint])
