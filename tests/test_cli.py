import json

import pytest

from airstriker_tensorflow.cli import main


def test_cli_train_evaluate_and_metadata(tmp_path, capsys):
    checkpoint = str(tmp_path / "agent.keras")
    main(["train", "--env", "synthetic", "--steps", "2", "--checkpoint", checkpoint])
    trained = json.loads(capsys.readouterr().out)
    assert trained["environment"] == "synthetic"
    assert trained["steps"] == 2
    assert trained["training"] is True
    metadata = json.loads((tmp_path / "agent.json").read_text())
    assert metadata["seed"] == 0
    assert metadata["environment"] == "synthetic"
    main(["evaluate", "--env", "synthetic", "--steps", "1", "--checkpoint", checkpoint])
    assert json.loads(capsys.readouterr().out)["training"] is False


def test_cli_rejects_missing_evaluation_checkpoint():
    with pytest.raises(SystemExit):
        main(["evaluate", "--env", "synthetic"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--env", "synthetic"],
        ["train", "--env", "synthetic", "--steps", "0", "--checkpoint", "unused.keras"],
    ],
)
def test_cli_rejects_invalid_training_configuration(arguments):
    with pytest.raises(SystemExit):
        main(arguments)


def test_failed_checkpoint_load_closes_environment(monkeypatch, tmp_path):
    from airstriker_tensorflow import cli
    from airstriker_tensorflow.environments import SyntheticEnvironment

    env = SyntheticEnvironment()
    closed = []
    monkeypatch.setattr(env, "close", lambda: closed.append(True))
    monkeypatch.setattr(cli, "SyntheticEnvironment", lambda: env)
    with pytest.raises(ValueError, match="not found"):
        main(["evaluate", "--env", "synthetic", "--checkpoint", str(tmp_path / "missing.keras")])
    assert closed == [True]


def test_default_game_is_the_included_airstriker_integration():
    from airstriker_tensorflow.cli import parse_args

    args = parse_args(["train", "--env", "retro", "--checkpoint", "agent.keras"])
    assert args.game == "Airstriker-Genesis-v0"
