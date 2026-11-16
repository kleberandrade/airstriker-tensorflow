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

## Longer training and fixed-budget comparison — 2026-11-16

- User explicitly requested longer real Airstriker training, comparison with random actions, new images, documentation and GitHub publication. Started from clean `72ffe4b`; saved original source/ROM hashes before running. Applied the established ub-mode/test-master workflow.
- Chose three independent fresh networks (seeds 7/13/29), 10,000 training transitions each, and 6,000 transitions per random/untrained/checkpoint evaluation. Retained the existing model, online update, preprocessing, 126-action space, epsilon and default savestate. The budget is 78.125× the original training smoke.
- Completed all 12 real emulator runs: 30,000 training and 54,000 evaluation/control transitions. Training rewards were `[860,920,900]`; random evaluation `[300,140,300]`; untrained evaluation `[460,0,180]`; checkpoint evaluation `[0,520,120]`. Checkpoint mean 213.33 versus random 246.67: one seed beat random, two lost. This sample does not establish an advantage.
- Audited 15 completed training episodes, 10 random-evaluation episodes, 10 untrained-evaluation episodes and five checkpoint-evaluation episodes. Kept unfinished final-episode returns separate; seed 13's 520 checkpoint points came entirely from an unfinished episode. Seed 7 repeated action 45 for all 6,000 greedy evaluation steps and earned zero.
- Wrote analysis tests before implementation and observed missing-module and schema assertion failures. Added exact trace accounting, compressed-trace reading, manifest mismatch rejection, complete/partial episode separation, positive-window validation and non-finite reward rejection. The algorithm running the game was not changed.
- Added optional Matplotlib 3.11.2 reporting support, regenerated the complete lock, and implemented a headless analysis example producing JSON plus static PNG/SVG figures. Verified the exported compressed traces reproduce the same analysis as the original raw data.
- Exported 296 native/processed gameplay PNGs, two PNG/two SVG figures, manifest, 12 losslessly compressed trace files, model/source/ROM provenance, local-checkpoint hashes and data-integrity hashes. Published no ROM or model checkpoint. Added four byte-identical selected images for README/report, including a successful seed and the failure.
- Expanded the README and teaching guide to explain fixed-budget comparisons, negative results, seed variation, complete versus unfinished episodes, and the distinction between training rewards and greedy evaluation. Kept the earlier short smoke report as historical evidence and warned against directly comparing its shorter-budget totals.
- Offline suite: 31 passed, 99% statement coverage; all six new analysis cases passed. Lint, formatting and pip consistency passed. Crapper maximum 8; no dryer candidates at 0.8. Mutator reported all 25 selected analysis mutations killed after rerunning with explicit all-site selection to avoid multiline LCOV mapping omissions.
- Captured and reviewed the two static plots and real successful/failed gameplay frames. Configured writable Matplotlib/font caches for this restricted local environment; no model warnings or bad outcomes were suppressed. Source hashes confirm the training/evaluation implementation stayed unchanged throughout collection.
- CI now additionally regenerates the longer experiment analysis/figures from committed compressed traces. Publication and its final CI result are reported in the completion message; no remote emulator training is claimed.
