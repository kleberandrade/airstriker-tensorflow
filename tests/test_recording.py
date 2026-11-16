"""Recorder tests use a synthetic fixture, never claim emulator validation."""

import json

import numpy as np
import pytest
from PIL import Image

from mario_tensorflow.environments import SyntheticEnvironment
from mario_tensorflow.recording import RecordedEnvironment


def test_recorder_preserves_transitions_and_saves_real_observations(tmp_path):
    env = RecordedEnvironment(SyntheticEnvironment(), tmp_path, capture_every=2)
    first, info = env.reset(seed=7)
    assert info == {"synthetic": True}
    assert env.action_count == 3
    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(1)
        assert reward == 1.0 and not terminated and not truncated
    env.close()
    records = [
        json.loads(line) for line in (tmp_path / "transitions.jsonl").read_text().splitlines()
    ]
    assert len(records) == 5
    assert records[0]["event"] == "reset"
    assert records[-1]["step"] == 4
    assert records[-1]["action"] == 1
    assert records[-1]["info"] == {"synthetic": True}
    captures = env.captures
    assert [c["step"] for c in captures] == [0, 1, 2, 4]
    np.testing.assert_array_equal(np.array(Image.open(tmp_path / captures[0]["rgb"])), first)
    np.testing.assert_array_equal(np.array(Image.open(tmp_path / captures[-1]["rgb"])), observation)
    assert Image.open(tmp_path / captures[-1]["processed"]).size == (32, 32)


def test_invalid_capture_interval_is_rejected_before_opening_output(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        RecordedEnvironment(SyntheticEnvironment(), tmp_path, capture_every=0)
    assert not (tmp_path / "transitions.jsonl").exists()


def test_numpy_info_scalars_are_serialized_and_unsupported_values_fail(tmp_path):
    class ScalarInfoEnvironment(SyntheticEnvironment):
        def step(self, action):
            image, reward, terminated, truncated, _ = super().step(action)
            return image, reward, terminated, truncated, {"score": np.int64(12)}

    recorded = RecordedEnvironment(ScalarInfoEnvironment(), tmp_path / "scalars")
    try:
        recorded.reset(seed=7)
        recorded.step(1)
    finally:
        recorded.close()
    last = json.loads((tmp_path / "scalars/transitions.jsonl").read_text().splitlines()[-1])
    assert last["info"]["score"] == 12

    class UnsupportedInfoEnvironment(SyntheticEnvironment):
        def reset(self, *, seed=None):
            image, _ = super().reset(seed=seed)
            return image, {"unsupported": object()}

    recorded = RecordedEnvironment(UnsupportedInfoEnvironment(), tmp_path / "unsupported")
    try:
        with pytest.raises(TypeError, match="Cannot serialize object"):
            recorded.reset(seed=7)
    finally:
        recorded.close()
