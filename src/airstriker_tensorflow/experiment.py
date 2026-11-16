"""Bounded emulator experiments with random and untrained control policies."""

import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from .agent import Agent
from .environments import DEFAULT_GAME, make_retro
from .recording import RecordedEnvironment
from .training import run


@dataclass(frozen=True)
class ExperimentConfig:
    game: str = DEFAULT_GAME
    seeds: tuple[int, ...] = (7, 13, 29)
    train_steps: int = 128
    evaluate_steps: int = 256
    capture_every: int = 128

    def __post_init__(self):
        if min(self.train_steps, self.evaluate_steps, self.capture_every) <= 0:
            raise ValueError("Step counts and capture interval must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide nonempty, distinct seeds")


class RandomPolicy:
    def __init__(self, action_count, seed):
        self.action_count = action_count
        self.rng = np.random.default_rng(seed)

    def act(self, state, epsilon):
        return int(self.rng.integers(self.action_count))


def _trial(config, output, seed, phase, env_factory, checkpoint=None):
    directory = f"seed-{seed}-{phase}"
    env = env_factory(config.game)
    try:
        recorder = RecordedEnvironment(env, output / directory, config.capture_every)
    except Exception:
        env.close()
        raise
    try:
        agent = _policy(phase, recorder.action_count, seed, checkpoint)
        started = perf_counter()
        training = phase == "train"
        steps = config.train_steps if training else config.evaluate_steps
        before = _weights(agent) if training else None
        result = run(recorder, agent, steps, seed=seed, training=training)
        result.update(
            phase=phase,
            seed=seed,
            elapsed_seconds=perf_counter() - started,
            directory=directory,
            action_count=recorder.action_count,
            captures=recorder.captures,
        )
        if training:
            result["weight_change_l2"] = float(np.linalg.norm(_weights(agent) - before))
            agent.save(output / f"seed-{seed}.keras")
    finally:
        recorder.close()
    result.update(_trace_summary(output / directory / "transitions.jsonl"))
    return result


def _policy(phase, action_count, seed, checkpoint):
    if phase == "random":
        return RandomPolicy(action_count, seed)
    return Agent(action_count, seed=seed, checkpoint=checkpoint)


def _weights(agent):
    return np.concatenate([weight.ravel() for weight in agent.model.get_weights()])


def _trace_summary(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    transitions = [record for record in records if record["event"] == "step"]
    actions = Counter(record["action"] for record in transitions)
    return {
        "unique_actions": len(actions),
        "action_counts": dict(actions),
        "nonzero_rewards": sum(record["reward"] != 0 for record in transitions),
        "terminations": sum(record["terminated"] for record in transitions),
        "truncations": sum(record["truncated"] for record in transitions),
        "final_info": transitions[-1]["info"],
    }


def execute(config, output, env_factory=make_retro):
    """Write one fresh experiment directory; existing evidence is never overwritten."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "configuration": asdict(config),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in ("tensorflow", "keras", "numpy", "Pillow", "stable-retro")
        },
        "runs": [],
    }
    for seed in config.seeds:
        for phase in ("random", "untrained", "train", "trained"):
            checkpoint = output / f"seed-{seed}.keras" if phase == "trained" else None
            manifest["runs"].append(_trial(config, output, seed, phase, env_factory, checkpoint))
        # Preserve completed seeds if a later emulator run fails.
        _save_manifest(output, manifest)
    return json.loads(json.dumps(manifest))


def _save_manifest(output, manifest):
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
