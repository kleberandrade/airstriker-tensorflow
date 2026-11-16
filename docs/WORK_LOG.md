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
