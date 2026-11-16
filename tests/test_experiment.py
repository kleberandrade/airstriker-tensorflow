"""Experiment orchestration tests use only the labeled synthetic fixture."""

import json

import pytest

from mario_tensorflow.environments import SyntheticEnvironment
from mario_tensorflow.experiment import ExperimentConfig, execute


def test_experiment_saves_results_and_is_explicitly_injected_offline(tmp_path):
    output = tmp_path / "experiment"
    config = ExperimentConfig(
        game="synthetic-fixture", seeds=(7,), train_steps=2, evaluate_steps=2, capture_every=2
    )
    manifest = execute(config, output, env_factory=lambda game: SyntheticEnvironment())
    assert manifest["configuration"]["game"] == "synthetic-fixture"
    assert [r["phase"] for r in manifest["runs"]] == ["random", "untrained", "train", "trained"]
    assert all(r["steps"] == 2 for r in manifest["runs"])
    training = manifest["runs"][2]
    assert training["weight_change_l2"] > 0
    assert (output / "seed-7.keras").exists()
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for run in manifest["runs"]:
        assert run["unique_actions"] >= 1
        assert (output / run["directory"] / "transitions.jsonl").exists()
        assert (output / run["directory"] / run["captures"][0]["rgb"]).exists()
    with pytest.raises(FileExistsError):
        execute(config, output, env_factory=lambda game: SyntheticEnvironment())


@pytest.mark.parametrize(
    "options",
    [
        {"train_steps": 0},
        {"evaluate_steps": 0},
        {"capture_every": 0},
        {"seeds": ()},
        {"seeds": (7, 7)},
    ],
)
def test_experiment_rejects_invalid_limits(options):
    with pytest.raises(ValueError):
        ExperimentConfig(**options)


def test_recorder_output_failure_closes_the_underlying_environment(tmp_path, monkeypatch):
    from mario_tensorflow import experiment

    closed = []

    class ClosingEnvironment(SyntheticEnvironment):
        def close(self):
            closed.append(True)

    def fail_recording(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(experiment, "RecordedEnvironment", fail_recording)
    with pytest.raises(OSError, match="disk unavailable"):
        execute(
            ExperimentConfig(seeds=(7,)),
            tmp_path / "failed",
            env_factory=lambda game: ClosingEnvironment(),
        )
    assert closed == [True]
