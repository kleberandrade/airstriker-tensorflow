# Mario TensorFlow

[![CI](https://github.com/kleberandrade/mario-tensorflow/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/kleberandrade/mario-tensorflow/actions/workflows/ci.yml)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Stable Retro 1.0.1](https://img.shields.io/badge/Stable_Retro-1.0.1-blue)
[![MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An educational experiment in teaching an agent to play **Super Mario World** with TensorFlow. It includes a small Keras Q-network, RGB preprocessing, bounded training, greedy evaluation, and an optional Stable Retro emulator adapter.

**Super Mario World gameplay has not been validated. All new emulator captures in this repository show Airstriker (Sega Genesis), not Mario (SNES). No trained Mario agent or Mario performance is claimed.** The offline smoke environment is synthetic test data, not a Mario simulation. The original experiment captured Snes9x windows and pressed random keys; its TensorFlow/training sections were unfinished.

## Airstriker integration test and detailed guide

A real **Airstriker** test now complements the synthetic smoke: three seeds, 2,688 recorded transitions, 90 captured PNGs, checkpoint reloads, and random/untrained/short-trained comparisons. Airstriker is the non-commercial test ROM included by Stable Retro; it is a different game from Mario. The 128-step training windows received no positive rewards, so the observed evaluation scores are exploratory integration results, not evidence of robust learning.

- [Detailed project guide](docs/PROJECT_GUIDE.md): architecture, data shapes, model, API, toolchain, setup, recording, migration and limitations.
- [Airstriker report with real screenshots](docs/experiments/2026-11-16-airstriker/REPORT.md): method, per-seed results, raw traces, package updates and provenance.

```sh
python examples/retro_experiment.py --output runs/airstriker-experiment
```

The experiment defaults to the included Airstriker ROM, seeds 7/13/29, 128 training steps and 256 evaluation steps per policy. It creates a new output directory and never silently switches to synthetic data. No ROM binary or checkpoint is committed.

## Requirements and installation

Python **3.13**, 64-bit Linux or macOS Apple Silicon, and a TensorFlow-compatible CPU. Local validation used macOS Apple Silicon; CI targets Ubuntu. Other Python versions, Intel macOS, GPU acceleration, and Windows have not been validated by this project.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
```

The lock records the complete development environment, including the optional emulator. For a smaller runtime installation, use `python -m pip install -e .`; add `'.[retro]'` for emulator support or `'.[dev,retro]'` for development. Direct requirements are pinned in `pyproject.toml`; the lock also pins transitive dependencies. No ROM is installed by this project.

## Quick start without a ROM

```sh
python examples/smoke.py
mario-tf train --env synthetic --steps 32 --seed 7 --checkpoint runs/synthetic.keras
mario-tf evaluate --env synthetic --steps 16 --seed 7 --checkpoint runs/synthetic.keras
```

The smoke example creates a temporary checkpoint, performs eight updates, evaluates for eight steps, and removes its output. JSON output reports steps, completed episodes, total reward and training loss. Synthetic episodes truncate after eight actions; only action 1 earns reward. Results verify plumbing, not gameplay skill.

## Real Super Mario World integration

Stable Retro replaces platform-specific desktop capture and keyboard injection with synchronized observations, actions, rewards, and episode boundaries. This is an explicit environment migration; TensorFlow remains the learning framework. See [upstream setup](https://stable-retro.farama.org/getting_started/) and the [API](https://stable-retro.farama.org/python/).

1. Obtain your own legally authorized Super Mario World ROM matching the integration hash. Owning a cartridge does not automatically establish copying rights in every jurisdiction. No ROM download service is recommended here.
2. Install the optional adapter and import only your authorized local files:

```sh
python -m pip install -e '.[retro]'
python -m stable_retro.import /absolute/path/to/your/authorized/roms
```

3. Confirm the integration and run a short experiment:

```sh
python -c "import stable_retro; print([g for g in stable_retro.data.list_games() if 'SuperMarioWorld' in g])"
mario-tf train --env retro --game SuperMarioWorld-Snes-v0 --steps 32 --seed 7 --checkpoint runs/mario.keras
mario-tf evaluate --env retro --game SuperMarioWorld-Snes-v0 --steps 32 --seed 7 --checkpoint runs/mario.keras
```

Use `--state NAME` for an installed integration savestate; otherwise Stable Retro uses its default. Rewards, termination rules, and savestates belong to the selected integration. Actions use its discrete filtered button combinations. No reward OCR or desktop-window detection is required. Missing ROM errors propagate; there is no automatic fallback to synthetic data.

**Mario-specific verified limitation:** Stable Retro 1.0.1 installed and listed `SuperMarioWorld-Snes-v0`, but creation failed with `FileNotFoundError: Game not found ... Did you make sure to import the ROM?`. No Super Mario World ROM was imported, downloaded, executed, or used for training. The separate Airstriker experiment below validates the shared adapter only. Adapter contract tests use a stub and do not validate emulation.

## Model, configuration, and API

RGB uint8 frames become normalized 32 × 32 grayscale images stacked across four observations. Reset fills the stack with the first frame to avoid mixing episodes. A small convolutional Keras model predicts one Q-value per discrete action. Online one-step Q-learning uses Adam (0.001), Huber loss, discount 0.99 and epsilon 0.1 during training; evaluation is greedy. True terminations stop bootstrapping; time-limit truncations retain it.

```python
from mario_tensorflow.agent import Agent
from mario_tensorflow.environments import SyntheticEnvironment
from mario_tensorflow.training import run

env = SyntheticEnvironment()  # Test fixture only
try:
    agent = Agent(env.action_count, seed=7)
    print(run(env, agent, steps=8, seed=7, training=True))
finally:
    env.close()
```

`--steps`, `--seed`, `--env`, `--game`, `--state`, and `--checkpoint` configure a run. `mario-tf --help` lists arguments. Training saves a `.keras` model with optimizer state and an adjacent `.json` run record. Training starts a new model; evaluation loads a checkpoint and checks its action count. Checkpoints are for evaluation, not exact training resumption: environment/RNG state is not restored. Use matching environment, game, and action mapping when evaluating.

## Development and validation

```sh
ruff check src tests examples
ruff format --check src tests examples
pytest --cov=mario_tensorflow --cov-report=term-missing --cov-report=lcov
python examples/smoke.py
python -m build
python -m pip check
```

Offline tests cover frame semantics, seeds, terminal targets, real weight updates, checkpoint roundtrip, unchanged evaluation weights, CLI records, and episode resets. CI installs the lock on Python 3.13, runs these checks, executes the smoke example and builds distributions. The CI badge links to the workflow on `master`; a local pass does not imply GitHub Actions has run the unpublished changes.

## Layout and migration

- `src/mario_tensorflow/`: environments, preprocessing, agent/model, train/evaluate loop, CLI.
- `tests/`: offline behavioral and adapter contract tests.
- `examples/`: executable offline smoke and real-emulator recording examples.
- `legacy/screen_capture.py`: unchanged historical `mario_bot_tensorflow.py`; archived desktop experiment, outside the supported toolchain.
- `docs/legacy-experiment.md`: historical notes; [work log](docs/WORK_LOG.md) records modernization and validation.
- `images/`: existing historical emulator/game screenshots.

Replace `python mario_bot_tensorflow.py` with the CLI commands above. The historical script remains available under `legacy/` for reference; its old GUI dependencies are intentionally excluded and it has not been revalidated. Historical screenshots and results are not evidence of the new agent's performance. Original random-key demonstration: [video](https://youtu.be/-JA13GCZLVQ). Thanks to Yuri Costa for recommending Snes9x.

## Limitations and license

This minimal baseline has no replay buffer, target network, frame skipping, tuned reward shaping, full-episode benchmark, or solved-game claim. Correlated online updates can be unstable; many more environment steps and algorithm improvements would be needed to investigate gameplay learning. Seeds reproduce local initial conditions but do not guarantee identical results across TensorFlow hardware/platforms. Long training never runs by default.

Project source code is [MIT licensed](LICENSE), retaining the original 2019 copyright. Nintendo game names, ROMs, and imagery retain their respective owners' rights and are not relicensed by this repository's MIT license. Stable Retro and emulator cores have [separate licenses](https://github.com/Farama-Foundation/stable-retro/blob/master/LICENSES.md). No ROM or pretrained weights are distributed.
