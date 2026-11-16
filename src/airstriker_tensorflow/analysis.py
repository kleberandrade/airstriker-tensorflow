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


def compare_evaluations(current, previous):
    keys = ("game", "seeds", "train_steps", "evaluate_steps")
    for key in keys:
        if current["configuration"][key] != previous["configuration"][key]:
            raise ValueError(f"Comparison requires the same protocol: {key}")
    random = _rewards_by_seed(current["groups"]["random"])
    untrained = _rewards_by_seed(current["groups"]["untrained"])
    online = _rewards_by_seed(previous["groups"]["trained"])
    dqn = _rewards_by_seed(current["groups"]["trained"])
    return {
        "protocol": {key: current["configuration"][key] for key in keys},
        "random_control_reproduced": random == _rewards_by_seed(previous["groups"]["random"]),
        "untrained_control_reproduced": untrained
        == _rewards_by_seed(previous["groups"]["untrained"]),
        "per_seed": [
            {
                "seed": seed,
                "random": random[seed],
                "untrained": untrained[seed],
                "online": online[seed],
                "dqn": dqn[seed],
                "dqn_minus_random": dqn[seed] - random[seed],
                "dqn_minus_online": dqn[seed] - online[seed],
            }
            for seed in current["configuration"]["seeds"]
        ],
        "mean_dqn_minus_random": current["groups"]["trained"]["mean_reward"]
        - current["groups"]["random"]["mean_reward"],
        "mean_dqn_minus_online": current["groups"]["trained"]["mean_reward"]
        - previous["groups"]["trained"]["mean_reward"],
    }
