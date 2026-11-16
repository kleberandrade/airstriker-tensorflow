"""Run a small real emulator experiment; defaults use the included Airstriker ROM."""

import argparse
import json
from pathlib import Path

from airstriker_tensorflow.cli import add_agent_arguments, agent_config_from_args
from airstriker_tensorflow.environments import DEFAULT_GAME
from airstriker_tensorflow.experiment import ExperimentConfig, execute


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default=DEFAULT_GAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 29])
    parser.add_argument("--train-steps", type=int, default=ExperimentConfig().train_steps)
    parser.add_argument("--evaluate-steps", type=int, default=256)
    parser.add_argument("--capture-every", type=int, default=128)
    add_agent_arguments(parser)
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        game=args.game,
        seeds=tuple(args.seeds),
        train_steps=args.train_steps,
        evaluate_steps=args.evaluate_steps,
        capture_every=args.capture_every,
        agent=agent_config_from_args(args),
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
