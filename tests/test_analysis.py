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
