"""Account for fixed-budget experiments without mixing complete and partial episodes."""

import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def summarize_trace(records, window=1000):
    if window <= 0:
        raise ValueError("window must be positive")
    steps = [row for row in records if row["event"] == "step"]
    rewards = [_reward(row) for row in steps]
    completed, unfinished = _episodes(steps)
    windows = [
        {
            "start": start + 1,
            "end": min(start + window, len(steps)),
            "steps": len(rewards[start : start + window]),
            "reward": math.fsum(rewards[start : start + window]),
        }
        for start in range(0, len(steps), window)
    ]
    return {
        "steps": len(steps),
        "reward": math.fsum(rewards),
        "completed_episodes": completed,
        "unfinished_episode": unfinished,
        "reward_windows": windows,
    }


def _reward(row):
    reward = float(row["reward"])
    if not math.isfinite(reward):
        raise ValueError("Trace rewards must be finite")
    return reward


def _episodes(steps):
    completed = []
    rewards = []
    for row in steps:
        rewards.append(float(row["reward"]))
        if row["terminated"] or row["truncated"]:
            completed.append(
                {
                    "return": math.fsum(rewards),
                    "length": len(rewards),
                    "ended_by": "terminated" if row["terminated"] else "truncated",
                }
            )
            rewards = []
    unfinished = {"return": math.fsum(rewards), "length": len(rewards)} if rewards else None
    return completed, unfinished


def _trace(folder):
    path = folder / "transitions.jsonl"
    opener = open
    if not path.exists():
        path = folder / "transitions.jsonl.gz"
        opener = gzip.open
    with opener(path, "rt") as stream:
        yield from (json.loads(line) for line in stream)


def _checked_run(directory, row, window):
    statistics = summarize_trace(_trace(directory / row["directory"]), window)
    if statistics["steps"] != row["steps"] or not math.isclose(
        statistics["reward"], row["reward"], rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError("Trace accounting differs from the manifest")
    result = dict(row, trace_statistics=statistics)
    result["finite_learning_metrics"] = all(
        math.isfinite(row[key]) for key in ("loss", "weight_change_l2") if row.get(key) is not None
    )
    return result


def _phase_statistics(runs):
    phases = sorted({row["phase"] for row in runs})
    result = {}
    for phase in phases:
        selected = [row for row in runs if row["phase"] == phase]
        rewards = [row["reward"] for row in selected]
        result[phase] = {
            "seeds": [row["seed"] for row in selected],
            "rewards": rewards,
            "mean_reward": mean(rewards),
            "seed_population_sd": pstdev(rewards),
            "completed_episodes": sum(
                len(row["trace_statistics"]["completed_episodes"]) for row in selected
            ),
        }
    return result


def _paired_difference(runs):
    random_rewards = {row["seed"]: row["reward"] for row in runs if row["phase"] == "random"}
    return [
        {"seed": row["seed"], "reward_difference": row["reward"] - random_rewards[row["seed"]]}
        for row in runs
        if row["phase"] == "trained" and row["seed"] in random_rewards
    ]


def analyze_experiment(directory, window=1000):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    runs = [_checked_run(directory, row, window) for row in manifest["runs"]]
    return {
        "algorithm": manifest.get("algorithm", "online-q-learning"),
        "configuration": manifest.get("configuration", {}),
        "source_manifest_sha256": hashlib.sha256(
            (directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "window_steps": window,
        "total_steps": sum(row["steps"] for row in runs),
        "groups": _phase_statistics(runs),
        "checkpoint_minus_random": _paired_difference(runs),
        "runs": runs,
    }


def _rewards_by_seed(group):
    return dict(zip(group["seeds"], group["rewards"], strict=True))


def _matching_protocol(current, previous, keys):
    for key in keys:
        if current["configuration"][key] != previous["configuration"][key]:
            raise ValueError(f"Comparison requires the same protocol: {key}")
    return {key: current["configuration"][key] for key in keys}


def _check_cohorts(current, previous):
    for experiment in (current, previous):
        expected = set(experiment["configuration"]["seeds"])
        for name in ("random", "untrained", "trained"):
            group = experiment["groups"][name]
            if len(group["seeds"]) != len(expected) or set(_rewards_by_seed(group)) != expected:
                raise ValueError("Comparison seed cohorts must contain every protocol seed once")


def _comparison_values(current, previous, before_label, after_label):
    _check_cohorts(current, previous)
    random = _rewards_by_seed(current["groups"]["random"])
    untrained = _rewards_by_seed(current["groups"]["untrained"])
    before = _rewards_by_seed(previous["groups"]["trained"])
    after = _rewards_by_seed(current["groups"]["trained"])
    return {
        "random_control_reproduced": random == _rewards_by_seed(previous["groups"]["random"]),
        "untrained_control_reproduced": untrained
        == _rewards_by_seed(previous["groups"]["untrained"]),
        "per_seed": [
            {
                "seed": seed,
                "random": random[seed],
                "untrained": untrained[seed],
                before_label: before[seed],
                after_label: after[seed],
                f"{after_label}_minus_random": after[seed] - random[seed],
                f"{after_label}_minus_{before_label}": after[seed] - before[seed],
            }
            for seed in current["configuration"]["seeds"]
        ],
        f"mean_{after_label}_minus_random": current["groups"]["trained"]["mean_reward"]
        - current["groups"]["random"]["mean_reward"],
        f"mean_{after_label}_minus_{before_label}": current["groups"]["trained"]["mean_reward"]
        - previous["groups"]["trained"]["mean_reward"],
    }


def compare_evaluations(current, previous):
    keys = ("game", "seeds", "train_steps", "evaluate_steps")
    return {
        "protocol": _matching_protocol(current, previous, keys),
        **_comparison_values(current, previous, "online", "dqn"),
    }


def compare_training_budgets(current, previous):
    """Compare budgets while requiring the same DQN and evaluation settings."""
    keys = ("game", "seeds", "evaluate_steps", "agent")
    protocol = _matching_protocol(current, previous, keys)
    if current["algorithm"] != previous["algorithm"] or current["algorithm"] != "dqn-replay-target":
        raise ValueError("Training-budget comparison requires the same DQN protocol")
    return {
        "comparison_type": "training_budget",
        "protocol": protocol,
        "training_steps": {
            "previous": previous["configuration"]["train_steps"],
            "current": current["configuration"]["train_steps"],
        },
        **_comparison_values(current, previous, "previous", "current"),
    }
