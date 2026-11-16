# Modernization log

- Inspected clean `master` at `aaeabb6`; original author is Kléber de Oliveira Andrade. No local changes or applicable AGENTS.md found.
- Applied ub-mode and test-master. Preserved MIT copyright and archived the original desktop experiment and notes; no third-party game files were present.
- The original never implemented a TensorFlow model or training. Add a deliberately small online Q-learning baseline, not a claimed trained Mario agent.
- Choose Python 3.13, TensorFlow 2.21 and optional Stable Retro 1.0.1 after checking upstream documentation and installing compatible Apple Silicon wheels. Replace brittle desktop title/key automation with an explicit Gymnasium emulator adapter; preserve history under legacy/.
- Plan offline behavioral tests for preprocessing, seeds, terminal bootstrap, actual parameter updates, checkpoint roundtrip, evaluation immutability, and bounded runs. The synthetic fixture is not Mario integration.

## Executed validation

- Tests were written before the new pipeline and CLI; observed import failures for missing modules, then passed after implementation. Expanded behavior coverage for terminal reset, deterministic initialization/actions, invalid CLI inputs and adapter options.
- `pytest --cov=mario_tensorflow --cov-report=term-missing --cov-report=lcov:target/coverage/python/lcov.info`: **14 passed**, 99% statement coverage (the module `__main__` line is exercised separately through CLI use). Coverage is offline only, not an emulator or trained-agent claim.
- `python examples/smoke.py`: eight synthetic train steps, checkpoint save/load, eight greedy evaluation steps. Parameter-update and evaluation-immutability tests use real TensorFlow.
- `ruff check src tests examples` and `ruff format --check src tests examples`: passed.
- `python -m build --no-isolation`: wheel and source distribution built; wheel installed under `/tmp/mario-wheel-check` and import resolved to that wheel. `pip check`: passed.
- Lock resolution dry-run for CPython 3.13 / Linux x86-64 binary wheels: passed. This does not execute Linux tests; GitHub Actions has not run these unpublished changes.
- `crapper --use-existing-coverage --threshold 9.99 src`: all functions below 10; maximum 8. Initially coverage was not found in pytest's default LCOV location; reran with the tool's supported `target/coverage/python/lcov.info` location. Split CLI parsing from resource lifecycle to reduce complexity.
- `dryer --threshold 0.8 src`: no duplicate candidates.
- `mutator --no-coverage --test-command '.venv/bin/python -m pytest tests/test_pipeline.py -q' --max-workers 2 src/mario_tensorflow/preprocessing.py`: all 11 selected mutants killed.
- Same mutation command selecting line 9 of `agent.py`: all five selected bootstrap-arithmetic mutants killed. Mutation checking was deliberately scoped to preprocessing and bootstrap arithmetic; no repository-wide mutation score is claimed.
- Stable Retro 1.0.1 imported and listed `SuperMarioWorld-Snes-v0`. Opening that integration returned `FileNotFoundError: Game not found ... import the ROM`. No ROM was obtained and no Mario integration/gameplay was executed. Adapter tests replace the factory only and are explicitly labeled contract tests.
- README and archived-document relative links/images checked: none missing. Original MIT notice retained; third-party ROM-download link removed; historical screenshot rights explicitly excluded from MIT relicensing.
- GitHub About updated successfully via authorized Git credential helper and confirmed by API readback. Exact applied text: **Educational TensorFlow Q-learning for Super Mario World, with reproducible offline tests and optional Stable Retro integration.**

## Limits and ownership

- Keras emits 12 NumPy `__array__(copy=...)` deprecation warnings during checkpoint serialization tests; tests pass. TensorFlow also reports CPU-frequency profiling warnings on this Mac. No warnings were suppressed.
- There is no replay/target network, trained model, benchmark, GPU test, desktop GUI test, or exact training-resume support. Synthetic results are test plumbing only.
- Unexpected edits to the historical-script path and packaging metadata were preserved. The user confirmed there is no other chat working on this project; inspected and adopted the final `legacy/screen_capture.py` and `pyproject.toml` contents into the final packaging commit. Historical script content matches the original exactly.
- No push, deployment, package publication, or long training was performed.

## Real emulator evidence and dependency refresh — 2026-11-16

- User requested a new ROM, all library updates, a real test, screenshots, collected results and detailed project documentation. Reused ub-mode and test-master; no push authorization was added.
- Searched upstream game/asset documentation. Selected Airstriker by Electrokinesis, already included in Stable Retro as a non-commercial test ROM. It is a Sega Genesis game, not Super Mario World. Verified its installed SHA-1 against the integration reference; copied no ROM into the project.
- Confirmed a clean Git tree before this round. Baseline: 14 tests passed. Queried all installed package versions and upgraded compatible pins with eager transitive resolution: NumPy 2.5.5, Pillow 13.0.0, pytest 9.2.0, Ruff 0.17.5, build 1.6.1 and setuptools 84.2.1. TensorFlow 2.21.0 and Stable Retro 1.0.1 remain current. Regenerated the complete lock.
- Preserved h5py 3.14.0 because TensorFlow requires `<3.15.0`, and pyglet 1.5.31 because Stable Retro requires `<2`. The remaining outdated-package report contains only these two constraints. Saved dependency metadata and before/after locks with the experiment evidence.
- Wrote recording/orchestration tests first and observed missing-module collection failures. Implemented a transparent recorder plus bounded multi-seed runner; tests verify exact PNG pixels, processed shape, JSON traces, output protection and actual TensorFlow updates/checkpoint creation using explicitly injected synthetic fixtures.
- Added resource/serialization error-path coverage. Ruff's newer diagnostics required import ordering and simple dict-literal cleanups; no algorithm rewrite was introduced.
- First real-run command safely refused to overwrite the directory already containing dependency audit files. Kept those files and used a new `run/` child directory; evidence was not overwritten.
- Executed 12 real emulator runs with seeds 7/13/29: random evaluation 256 steps, untrained evaluation 256, training 128, checkpoint evaluation 256 per seed. Total 2,688 transitions; 45 capture points and 90 native/processed PNGs. Visually inspected actual random and checkpoint-evaluation frames; screenshots show Airstriker game state and score, not synthetic observations.
- Measured reward vectors: random `[0,0,0]`, untrained `[20,0,0]`, train `[0,0,0]`, checkpoint evaluation `[20,40,40]`. All training rewards were zero; weight changes therefore reflect bootstrap updates, and the small evaluation difference is not evidence of robust learning. No full episode completed. The report states those limits explicitly.
- Saved raw per-step JSONL traces, manifest, source/artifact hashes, ROM/integration provenance, diagnostic logs and local ignored checkpoints. Added `docs/PROJECT_GUIDE.md` and `docs/experiments/2026-11-16-airstriker/REPORT.md`, and linked them from README. Preserved MIT code notices and explicitly retained third-party game-image rights.
- Offline suite: 24 passed, 97% statement coverage, 18 unsuppressed upstream Keras/NumPy deprecation warnings. Real emulator execution is recorded separately; offline coverage is not claimed as ROM integration coverage.
- Lint/format, pip consistency, package builds and Linux CPython 3.13 binary-resolution dry-run passed. No Linux runtime, GPU, Mario gameplay or remote CI execution is claimed. TensorFlow CPU-profiling diagnostics remain in the run log.
- Crapper maximum 8, every source function below 10; dryer found no candidates at 0.8. Recorder mutation check reported all 24 selected mutants killed; this scope is distinct from the earlier preprocessing/target checks and is not a project-wide mutation score.
- No long training, ROM redistribution, push, deployment or package publication was performed.
- Final artifact checks: zero broken relative documentation/image links; all 106 recorded artifact hashes matched; manifest/trace totals matched 2,688 transitions and 90 PNGs. Installed the new wheel into a separate temporary target and successfully ran its synthetic train/save/load/evaluate smoke. Source archive includes the guide, manifest and all 90 screenshots, with no ROM or `.keras` files.
