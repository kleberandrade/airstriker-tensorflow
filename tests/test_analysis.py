"""Exact accounting tests on invented traces; no gameplay claims."""

import gzip
import json

import pytest

from airstriker_tensorflow.analysis import analyze_experiment, summarize_trace


def test_episode_returns_keep_the_unfinished_tail_separate():
    records = [
        {"event": "reset"},
        {"event": "step", "reward": 2.0, "terminated": False, "truncated": False},
        {"event": "step", "reward": 3.0, "terminated": True, "truncated": False},
        {"event": "reset"},
        {"event": "step", "reward": 4.0, "terminated": False, "truncated": True},
        {"event": "reset"},
        {"event": "step", "reward": -1.0, "terminated": False, "truncated": False},
    ]
    result = summarize_trace(records, window=3)
    assert result["steps"] == 4
    assert result["reward"] == 8.0
    assert result["completed_episodes"] == [
        {"return": 5.0, "length": 2, "ended_by": "terminated"},
        {"return": 4.0, "length": 1, "ended_by": "truncated"},
    ]
    assert result["unfinished_episode"] == {"return": -1.0, "length": 1}
    assert result["reward_windows"] == [
        {"start": 1, "end": 3, "steps": 3, "reward": 9.0},
        {"start": 4, "end": 4, "steps": 1, "reward": -1.0},
    ]


def test_empty_and_exactly_completed_trace_have_no_unfinished_episode():
    assert summarize_trace([], window=2)["unfinished_episode"] is None
    row = {"event": "step", "reward": 1.0, "terminated": True, "truncated": False}
    assert summarize_trace([row], window=2)["unfinished_episode"] is None


def test_analysis_reads_compressed_traces_and_compares_matched_seeds(tmp_path):
    runs = []
    for phase, reward in [("random", 3.0), ("trained", 5.0)]:
        directory = f"seed-7-{phase}"
        folder = tmp_path / directory
        folder.mkdir()
        with gzip.open(folder / "transitions.jsonl.gz", "wt") as stream:
            stream.write(
                json.dumps(
                    {"event": "step", "reward": reward, "terminated": False, "truncated": False}
                )
                + "\n"
            )
        runs.append(
            {
                "directory": directory,
                "phase": phase,
                "seed": 7,
                "steps": 1,
                "reward": reward,
                "loss": None,
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    result = analyze_experiment(tmp_path, window=1000)
    assert result["groups"]["trained"]["mean_reward"] == 5.0
    assert result["checkpoint_minus_random"] == [{"seed": 7, "reward_difference": 2.0}]
    assert result["total_steps"] == 2
    runs[0]["reward"] = 9.0
    (tmp_path / "manifest.json").write_text(json.dumps({"runs": runs}))
    with pytest.raises(ValueError, match="manifest"):
        analyze_experiment(tmp_path)


@pytest.mark.parametrize("window", [0, -1])
def test_bad_window_is_rejected(window):
    with pytest.raises(ValueError, match="positive"):
        summarize_trace([], window=window)


def test_nonfinite_reward_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        summarize_trace(
            [{"event": "step", "reward": float("nan"), "terminated": False, "truncated": False}]
        )


def test_algorithm_comparison_requires_matching_protocol_and_discloses_control_changes():
    from airstriker_tensorflow.analysis import compare_evaluations

    config = {
        "game": "Airstriker-Genesis-v0",
        "seeds": [7, 13, 29],
        "train_steps": 10000,
        "evaluate_steps": 6000,
    }

    def summary(trained):
        return {
            "configuration": dict(config),
            "groups": {
                "random": {
                    "seeds": [7, 13, 29],
                    "rewards": [300.0, 140.0, 300.0],
                    "mean_reward": 740 / 3,
                },
                "untrained": {
                    "seeds": [7, 13, 29],
                    "rewards": [460.0, 0.0, 180.0],
                    "mean_reward": 640 / 3,
                },
                "trained": {
                    "seeds": [7, 13, 29],
                    "rewards": trained,
                    "mean_reward": sum(trained) / 3,
                },
            },
        }

    old = summary([0.0, 520.0, 120.0])
    new = summary([100.0, 600.0, 200.0])  # invented values; not a real result
    result = compare_evaluations(new, old)
    assert result["random_control_reproduced"] is True
    assert result["untrained_control_reproduced"] is True
    assert result["per_seed"][0]["dqn_minus_online"] == 100.0
    assert result["mean_dqn_minus_random"] == pytest.approx(160 / 3)
    new["groups"]["untrained"]["rewards"][0] = 0.0
    assert compare_evaluations(new, old)["untrained_control_reproduced"] is False
    new["configuration"]["evaluate_steps"] = 256
    with pytest.raises(ValueError, match="protocol"):
        compare_evaluations(new, old)


def test_training_budget_comparison_keeps_dqn_and_evaluation_protocol_fixed():
    from airstriker_tensorflow.analysis import compare_training_budgets

    config = {
        "game": "Airstriker-Genesis-v0",
        "seeds": [7, 13, 29],
        "train_steps": 10000,
        "evaluate_steps": 6000,
        "agent": {
            "replay_capacity": 5000,
            "batch_size": 32,
            "learning_starts": 256,
            "target_sync_interval": 250,
        },
    }

    def result(train_steps, rewards):
        cfg = dict(config, train_steps=train_steps)

        def group(values):
            return {
                "seeds": [7, 13, 29],
                "rewards": values,
                "mean_reward": sum(values) / len(values),
            }

        return {
            "algorithm": "dqn-replay-target",
            "configuration": cfg,
            "groups": {
                "random": group([300.0, 140.0, 300.0]),
                "untrained": group([460.0, 0.0, 180.0]),
                "trained": group(rewards),
            },
        }

    previous = result(10000, [640.0, 0.0, 520.0])
    current = result(100000, [100.0, 200.0, 300.0])  # invented fixture, not measured results
    comparison = compare_training_budgets(current, previous)
    assert comparison["training_steps"] == {"previous": 10000, "current": 100000}
    assert comparison["random_control_reproduced"] is True
    assert comparison["untrained_control_reproduced"] is True
    assert comparison["per_seed"][0]["current_minus_previous"] == -540.0
    assert comparison["mean_current_minus_previous"] == pytest.approx(-560 / 3)
    current["configuration"]["evaluate_steps"] = 256
    with pytest.raises(ValueError, match="protocol"):
        compare_training_budgets(current, previous)
    current["configuration"]["evaluate_steps"] = 6000
    current["configuration"]["agent"] = dict(config["agent"], batch_size=16)
    with pytest.raises(ValueError, match="protocol"):
        compare_training_budgets(current, previous)
    current["configuration"]["agent"] = config["agent"]
    current["algorithm"] = "online-q-learning"
    with pytest.raises(ValueError, match="DQN"):
        compare_training_budgets(current, previous)


@pytest.mark.parametrize("changed", ["game", "seeds", "agent", "algorithm"])
def test_budget_comparison_rejects_other_protocol_changes(changed):
    from copy import deepcopy

    from airstriker_tensorflow.analysis import compare_training_budgets

    previous = {
        "algorithm": "dqn-replay-target",
        "configuration": {
            "game": "Airstriker-Genesis-v0",
            "seeds": [7, 13, 29],
            "train_steps": 10000,
            "evaluate_steps": 6000,
            "agent": {
                "replay_capacity": 5000,
                "batch_size": 32,
                "learning_starts": 256,
                "target_sync_interval": 250,
            },
        },
    }
    current = deepcopy(previous)
    current["configuration"]["train_steps"] = 100000
    if changed == "algorithm":
        current["algorithm"] = "online-q-learning"
    elif changed == "agent":
        current["configuration"]["agent"]["batch_size"] = 16
    elif changed == "seeds":
        current["configuration"]["seeds"] = [1, 2, 3]
    else:
        current["configuration"]["game"] = "different-game"
    with pytest.raises(ValueError, match="protocol"):
        compare_training_budgets(current, previous)


def test_budget_comparison_rejects_an_incomplete_seed_cohort():
    from copy import deepcopy

    from airstriker_tensorflow.analysis import compare_training_budgets

    group = {"seeds": [7, 13, 29], "rewards": [1.0, 2.0, 3.0], "mean_reward": 2.0}
    previous = {
        "algorithm": "dqn-replay-target",
        "configuration": {
            "game": "Airstriker-Genesis-v0",
            "seeds": [7, 13, 29],
            "train_steps": 10000,
            "evaluate_steps": 6000,
            "agent": {},
        },
        "groups": {name: deepcopy(group) for name in ("random", "untrained", "trained")},
    }
    current = deepcopy(previous)
    current["configuration"]["train_steps"] = 100000
    current["groups"]["trained"] = {"seeds": [7], "rewards": [1.0], "mean_reward": 1.0}
    with pytest.raises(ValueError, match="cohort"):
        compare_training_budgets(current, previous)
