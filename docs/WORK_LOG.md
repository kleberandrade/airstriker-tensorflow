# Airstriker TensorFlow work log

## Preserved experiment evidence

- The real Stable Retro experiment uses the upstream non-commercial Airstriker test ROM; its installed SHA-1 matches `a8be7b0ca850119b167f9644e6a4a22e983d61a4`.
- Three seeds (7, 13, 29), four phases per seed, 2,688 transitions and 90 PNGs are preserved under `docs/experiments/2026-11-16-airstriker/`.
- Recorded reward vectors: random `[0,0,0]`, untrained `[20,0,0]`, train `[0,0,0]`, checkpoint evaluation `[20,40,40]`. No completed episode or robust-learning claim is made.
- Raw transitions, manifest, images and checkpoint hashes are preserved. Source hashes describe the original recorded source bytes, identified by module basename after the package migration; they are not recomputed as though the current source produced historical measurements.
- Earlier development history remains in Git. This current log focuses on the supported Airstriker project; original MIT copyright and third-party asset rights remain intact.

## Airstriker-only migration — 2026-11-16

- User requested a complete project conversion, educational documentation, game images and GitHub updates. Applied ub-mode and test-master.
- Started from clean `ec0afc0`. Wrote a regression test that failed because the default CLI game differed from Airstriker, then changed the default to the included integration.
- Renamed the distribution to `airstriker-tensorflow` (0.3.0), Python package to `airstriker_tensorflow`, CLI to `airstriker-tf`, and local directory to `/Users/kleber/Projects/airstriker-tensorflow`. Centralized the default integration in `DEFAULT_GAME`.
- Migrated imports, examples, test coverage configuration and package metadata. Recreated the virtual environment at the new location from the existing pinned lock to avoid stale absolute paths.
- Removed unsupported desktop code, obsolete game-specific notes and unrelated screenshots from the active tree. Their content remains recoverable from Git history; no history was rewritten.
- Added README images copied byte-for-byte from the actual recorded Airstriker frames, with capture context and a processed-input preview. No generated imagery, ROM or checkpoint is committed.
- Rewrote the README and teaching guide around ten practical lessons: environment inspection, terminology, pixel preprocessing, model shapes, numeric bootstrap targets, exploration/evaluation, controlled comparisons, trace inspection, code boundaries and experiment design. Included exercises, expected outputs and honest limits.
- Retained original gameplay evidence and refreshed current test/build/tool reports rather than rewriting the meaning of old measurements. Corrected repository URLs and retained upstream licensing/provenance links.
- GitHub repository was renamed to `kleberandrade/airstriker-tensorflow`; updated origin, About description and topics (Airstriker, TensorFlow, Keras, reinforcement learning, Q-learning, Stable Retro, Python and education). The default branch remains `master`.
- Migration validation: 25 offline tests passed, 99% statement coverage; lint, formatting, pinned dependency consistency and package builds passed. The new CLI completed 16 real training steps and 32 evaluation steps without a game override, proving the Airstriker default.
- Re-ran the complete default three-seed Airstriker experiment in an ignored output directory and executed all six Python tutorial blocks in order against the real emulator. Their shapes, API calls, trace analysis, target calculations and training/evaluation examples succeeded.
- Checked all relative Markdown/image links, preserved original 106 artifact hashes, and verified that three README/tutorial images are byte-identical to their recorded source frames. Package archives contain 90 recorded PNGs plus three teaching images and exclude ROMs/checkpoints.
- Refreshed test/build/complexity/mutation logs under the new package name. Cleared obsolete mutation snapshots before checking the renamed source: all 24 selected recorder mutations killed; maximum CRAP 8; dryer found no candidates at 0.8.
- Searched the maintained source, examples, tests, configuration, docs and image names for superseded game/project references; no active reference remains. Git history and original recorded numerical/gameplay evidence were not rewritten.
