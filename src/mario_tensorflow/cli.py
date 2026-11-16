"""Command-line entry point; emulator access always requires an explicit choice."""

import argparse
import json
from pathlib import Path

from .agent import Agent
from .environments import SyntheticEnvironment, make_retro
from .training import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--env", required=True, choices=["synthetic", "retro"])
    parser.add_argument("--game", default="SuperMarioWorld-Snes-v0")
    parser.add_argument("--state", help="Stable Retro savestate name; default from integration")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    if args.mode == "train" and not args.checkpoint:
        parser.error("train requires --checkpoint output path")
    return args


def main(argv=None):
    args = parse_args(argv)
    env = SyntheticEnvironment() if args.env == "synthetic" else make_retro(args.game, args.state)
    try:
        checkpoint = args.checkpoint if args.mode == "evaluate" else None
        agent = Agent(env.action_count, seed=args.seed, checkpoint=checkpoint)
        result = run(env, agent, steps=args.steps, seed=args.seed, training=args.mode == "train")
        metadata = {
            "environment": args.env,
            "game": args.game if args.env == "retro" else None,
            "state": args.state,
            "seed": args.seed,
            "training": args.mode == "train",
        }
        result.update(metadata)
        if args.mode == "train":
            agent.save(args.checkpoint)
            args.checkpoint.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        env.close()


if __name__ == "__main__":
    main()
