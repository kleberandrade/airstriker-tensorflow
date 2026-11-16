"""Command-line entry point; emulator access always requires an explicit choice."""

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

from .agent import ALGORITHM_NAME, Agent, DQNConfig
from .environments import DEFAULT_GAME, SyntheticEnvironment, make_retro
from .training import run


def add_agent_arguments(parser):
    defaults = DQNConfig()
    for field in fields(DQNConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"), type=int, default=getattr(defaults, field.name)
        )


def agent_config_from_args(args):
    return DQNConfig(**{field.name: getattr(args, field.name) for field in fields(DQNConfig)})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--env", required=True, choices=["synthetic", "retro"])
    parser.add_argument("--game", default=DEFAULT_GAME)
    parser.add_argument("--state", help="Stable Retro savestate name; default from integration")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    add_agent_arguments(parser)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    if args.mode == "train" and not args.checkpoint:
        parser.error("train requires --checkpoint output path")
    try:
        args.agent_config = agent_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    env = SyntheticEnvironment() if args.env == "synthetic" else make_retro(args.game, args.state)
    try:
        checkpoint = args.checkpoint if args.mode == "evaluate" else None
        agent = Agent(
            env.action_count, seed=args.seed, checkpoint=checkpoint, config=args.agent_config
        )
        result = run(env, agent, steps=args.steps, seed=args.seed, training=args.mode == "train")
        metadata = {
            "environment": args.env,
            "game": args.game if args.env == "retro" else None,
            "state": args.state,
            "seed": args.seed,
            "training": args.mode == "train",
            "algorithm": ALGORITHM_NAME,
            "agent_config": asdict(agent.config),
            "optimizer_updates": agent.updates,
            "target_syncs": agent.target_syncs,
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
