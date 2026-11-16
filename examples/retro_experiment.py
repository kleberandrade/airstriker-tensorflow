"""Run a small real emulator experiment; defaults use the included Airstriker ROM."""

import argparse
import json
from pathlib import Path

from mario_tensorflow.experiment import ExperimentConfig, execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="Airstriker-Genesis-v0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 29])
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--evaluate-steps", type=int, default=256)
    parser.add_argument("--capture-every", type=int, default=128)
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        game=args.game,
        seeds=tuple(args.seeds),
        train_steps=args.train_steps,
        evaluate_steps=args.evaluate_steps,
        capture_every=args.capture_every,
    )
    result = execute(config, args.output)
    print(
        json.dumps(
            [
                {"seed": r["seed"], "phase": r["phase"], "reward": r["reward"], "steps": r["steps"]}
                for r in result["runs"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
