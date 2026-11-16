# Project guide

## Purpose and current status

Mario TensorFlow is an educational reinforcement-learning project. It explores how RGB observations, button combinations, environment rewards and a small TensorFlow Q-network fit together. The original 2019 experiment automated a Snes9x window with random keyboard presses. The maintained implementation uses a synchronous Stable Retro adapter and implements the previously unfinished learning loop.

The primary research subject remains Super Mario World. The reproducible real-emulator test uses **Airstriker**, a different Sega Genesis game included by Stable Retro for non-commercial testing. It proves that the adapter, image pipeline, training updates, checkpoint loading and evaluation work together on an actual emulator. It does not prove that the agent can play Mario.

See the [recorded experiment](experiments/2026-11-16-airstriker/REPORT.md), including screenshots, exact configuration, raw transitions and limits. No Mario ROM, trained Mario policy or solved-game claim is provided.

## Supported toolchain

The project deliberately supports CPython **3.13**. Validation took place on macOS Apple Silicon with CPU execution; GitHub Actions targets Ubuntu. Linux package resolution was checked, but this local run did not execute on Linux. GPU execution, native Windows and Intel macOS have not been validated.

The current direct runtime pins are TensorFlow 2.21.0, NumPy 2.5.5 and Pillow 13.0.0. The optional emulator is Stable Retro 1.0.1. Development pins are pytest 9.2.0, pytest-cov 7.1.0, Ruff 0.17.5, build 1.6.1 and setuptools 84.2.1.

All installed dependencies were checked against the package index. Two transitive packages remain below their newest available releases because of upstream constraints: TensorFlow requires `h5py<3.15.0` (installed 3.14.0), and Stable Retro requires `pyglet<2` (installed 1.5.31). Forcing upgrades would violate those packages' declared compatibility. [Dependency evidence](experiments/2026-11-16-airstriker/dependency-updates.json) records the old and new versions; the complete environment is pinned in `requirements-dev.lock`.

## Installation

From the repository root:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m pip check
```

This installs the tested development environment, including Stable Retro. To install only the runtime, use `python -m pip install -e .`. For runtime plus emulator, use `python -m pip install -e '.[retro]'`. Extras are installed by pip; no game download command is executed by project code.

To update dependencies later, check `python -m pip list --outdated --format=json`, update compatible direct pins in `pyproject.toml`, install with `python -m pip install --upgrade --upgrade-strategy eager -e '.[dev,retro]'`, regenerate the lock using `python -m pip freeze --exclude mario-tensorflow > requirements-dev.lock`, and repeat the validations below. Do not remove upstream constraints merely to eliminate an outdated-package report.

## Source layout and responsibilities

- `src/mario_tensorflow/environments.py`: explicitly labeled synthetic fixture and the optional real emulator factory.
- `preprocessing.py`: image boundary validation, RGB-to-grayscale conversion, resizing, normalization and four-frame history.
- `agent.py`: Keras Q-network, seeded epsilon-greedy policy, bootstrap target, optimizer update and checkpoint operations.
- `training.py`: bounded environment interaction shared by training and greedy evaluation.
- `cli.py`: user arguments, checkpoint/run metadata and guaranteed environment cleanup.
- `recording.py`: transparent environment wrapper that writes real RGB frames, processed previews and transition traces.
- `experiment.py`: bounded random/untrained/train/trained comparisons across explicit seeds.
- `examples/smoke.py`: temporary offline train/save/load/evaluate example.
- `examples/retro_experiment.py`: reproducible real-emulator experiment command.
- `tests/`: offline behavioral tests. Tests that inject synthetic environments or replace a factory are labeled accordingly.
- `legacy/screen_capture.py`, `docs/legacy-experiment.md`, `images/`: preserved original experiment and historical imagery.

## Data flow and environment contract

The real factory calls `stable_retro.make` with RGB rendering and `Actions.DISCRETE`. It reads the resulting action-space size rather than hardcoding a controller layout. Airstriker currently exposes 126 filtered discrete button combinations. Each action advances one environment step; there is no frame skipping or action-repeat wrapper.

The common environment contract is:

```python
observation, info = env.reset(seed=7)
observation, reward, terminated, truncated, info = env.step(action_index)
env.close()
```

Observations must be `uint8` RGB arrays with shape `(height, width, 3)`. The preprocessor converts them to grayscale and resizes to 32×32. Dividing by 255 produces `float32` values in `[0, 1]`. Four successive frames form `(32, 32, 4)`, occupying 16 KiB per state. At reset all four channels contain the first observation, preventing the previous episode from leaking into the new one.

Airstriker's actual captured observations were 224×320 RGB. The processed PNG is a visualization of the newest grayscale channel, not all four model channels and not an invented game image.

A true terminal prevents bootstrapping; a time-limit truncation ends the episode but still permits bootstrapping from the final observation. The loop resets after either flag and increments the reset seed by completed-episode count. The caller closes the environment; the CLI and experiment runner use `finally` for this lifecycle.

## Model and update rule

The model is intentionally small:

1. Input `(32, 32, 4)`.
2. Eight 5×5 convolution filters, stride 4, ReLU.
3. Flatten the resulting 7×7×8 tensor.
4. Dense layer with 32 ReLU units.
5. One linear Q-value per environment action.

Airstriker's 126-action model contains **17,542 trainable parameters**. Adam uses learning rate 0.001 and Huber loss. Training chooses a random action with probability 0.1; otherwise it uses the maximum predicted Q-value. Evaluation always uses the maximum Q-value.

For each observed transition, the selected action's target is:

```text
target = reward + 0.99 × (1 - terminated) × max(Q(next_state))
```

Other action targets retain their current predicted values. One optimizer update follows each training transition. This is online one-step Q-learning, without replay, a target network, Double DQN, exploration scheduling or a tuned reward function. Its correlated updates can be unstable. Nonzero weight changes with zero rewards can result from bootstrapping; they are not evidence of learning a useful task.

Seeds initialize Python/NumPy/TensorFlow through Keras and the agent's own action generator. The environment receives an explicit seed. This supports repeatable local experiments, not guaranteed bit-identical results across processors, package versions or TensorFlow kernels.

## CLI usage

The environment choice is mandatory; missing ROM errors never silently select the synthetic fixture.

```sh
mario-tf --help
mario-tf train --env synthetic --steps 32 --seed 7 --checkpoint runs/synthetic.keras
mario-tf evaluate --env synthetic --steps 16 --seed 7 --checkpoint runs/synthetic.keras

mario-tf train --env retro --game Airstriker-Genesis-v0 --steps 128 --seed 7 --checkpoint runs/airstriker.keras
mario-tf evaluate --env retro --game Airstriker-Genesis-v0 --steps 256 --seed 7 --checkpoint runs/airstriker.keras
```

`--state NAME` selects an integration savestate. Omitting it uses the integration's default. `--steps` is a hard interaction bound and must be positive. `--checkpoint` is required: an output path in training mode and an input model in evaluation mode. Training starts a new network; it does not resume an existing checkpoint.

Training writes the `.keras` model and a neighboring `.json` record. The record includes environment kind, game, selected state, seed, training flag, total reward, step count, completed episodes and average training loss. Evaluation does not modify model weights. Checkpoint loading checks action count, but not the identity/order of controller actions: use the same game and integration when evaluating.

A Keras file contains model and optimizer state. Environment state, action-generator state and frame history are not restored. The current API therefore supports inference/evaluation checkpoints, not exact interrupted-training recovery.

## Python API

```python
from mario_tensorflow.agent import Agent
from mario_tensorflow.environments import make_retro
from mario_tensorflow.training import run

env = make_retro("Airstriker-Genesis-v0")
try:
    agent = Agent(env.action_count, seed=7)
    training = run(env, agent, steps=128, seed=7, training=True)
    agent.save("runs/airstriker.keras")
    evaluation = run(env, agent, steps=256, seed=7, training=False)
    print(training, evaluation)
finally:
    env.close()
```

`run` returns `steps`, completed `episodes`, cumulative `reward` and mean `loss` (`None` when evaluating). Reward and episode definitions come from the integration, not the learning code. Mean loss is a diagnostic of the optimizer API, not a performance or convergence metric.

## Reproducible real-emulator recording

```sh
python examples/retro_experiment.py \
  --output runs/airstriker-experiment \
  --seeds 7 13 29 \
  --train-steps 128 \
  --evaluate-steps 256 \
  --capture-every 128
```

The output directory must be new. This guard prevents accidental destruction of collected evidence. For each seed, the runner evaluates a seeded random policy, evaluates an untrained network, trains a fresh network with the same initialization seed, saves it, and evaluates the loaded checkpoint. Every phase resets the same game/state/seed for a comparable starting condition.

Each phase writes `transitions.jsonl`, RGB PNG captures and processed grayscale PNGs. Frames are sampled at reset, step 1 and the configured interval. The manifest includes package/host versions, configuration, rollout timings, total rewards, loss, action histogram, nonzero-reward counts, termination/truncation counts, final integration variables and capture paths. Completed seeds are saved even if a later seed fails. Local `.keras` checkpoints are ignored by Git.

Elapsed times include rollout, inference/updates, trace writes and screenshots. They exclude model construction/checkpoint loading and final checkpoint saving. Treat them as execution observations on one machine, not an emulator throughput benchmark.

## Mario assets and migration

For Super Mario World, supply a legally authorized local ROM matching the Stable Retro integration hash. The supported integration name is `SuperMarioWorld-Snes-v0`.

```sh
python -m stable_retro.import /absolute/path/to/your/authorized/roms
mario-tf train --env retro --game SuperMarioWorld-Snes-v0 --steps 128 --seed 7 --checkpoint runs/mario.keras
```

The recorded test did not obtain or execute a Nintendo ROM. Airstriker is a testing alternative and is never silently substituted for Mario. See [Stable Retro setup](https://stable-retro.farama.org/getting_started/) and [upstream included test ROMs](https://github.com/Farama-Foundation/stable-retro/blob/main/docs/included_roms.md).

The old `python mario_bot_tensorflow.py` entry point moved unchanged to `legacy/screen_capture.py`. Use `mario-tf` for supported runs. The legacy script's window title, keyboard control, Canny preprocessing and GUI dependencies are historical and were not revalidated. No old screenshots are presented as evidence of the current implementation.

## Validation and troubleshooting

```sh
ruff check src tests examples
ruff format --check src tests examples
pytest --cov=mario_tensorflow --cov-report=term-missing --cov-report=lcov:target/coverage/python/lcov.info
python examples/smoke.py
python -m pip check
python -m build --no-isolation
```

The suite checks image shape/range/history, invalid inputs, seeded actions and parameters, actual weight updates, terminal targets, episode reset, checkpoint equality, evaluation immutability, CLI configuration, adapter options, trace/image fidelity and experiment orchestration. Most tests are offline. The separate Airstriker execution is the real integration evidence.

- `Game not found ... import the ROM`: the integration is listed but its required local ROM is absent or does not match. There is no project-managed commercial ROM downloader.
- `FileExistsError` for experiment output: choose a new directory to preserve previous evidence.
- Checkpoint action mismatch: select a checkpoint from the same game/integration. An equal action count alone cannot guarantee compatible action semantics.
- Keras/NumPy checkpoint warnings: the tested stack emits `__array__(copy=...)` deprecation warnings. They remain visible; model roundtrip tests pass.
- TensorFlow CPU profiling warnings on this Mac: recorded as emitted; no GPU was available or used.
- Unavailable emulator wheel: consult upstream platform build instructions. A successful pip resolution on another platform is not a substitute for running its tests.

GitHub CI installs the lock, runs offline tests, lint, formatting, the synthetic smoke example and package build. It does not currently execute the recorded Airstriker experiment or verify Mario. These changes have not been pushed, so no remote CI success is claimed.

## Interpretation, limitations and rights

Three seeds and a few hundred steps are a smoke experiment. The recorded training windows received no positive reward, even though weights changed and later evaluation scores differed. There is no statistical confidence interval, solved level, full-episode benchmark, long training or validated Mario transfer. The severe 32×32 downsampling and broad action space also limit this baseline.

Project code remains MIT with the original copyright. Game imagery and ROM content retain their owners' rights. Airstriker is supplied upstream as a non-commercial test ROM; this project does not claim it is public domain or relicense its screenshots as MIT. No ROM binary is copied into the repository. Stable Retro and emulator cores retain [their separate licenses](https://github.com/Farama-Foundation/stable-retro/blob/main/LICENSES.md).
