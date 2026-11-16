# ALFWorld 30-Case Paired Comparison Implementation Plan

> **For agentic workers:** execute inline with test-driven development; do not contact other sessions.

**Goal:** Run the same 30 fixed ALFWorld development games with and without the static action guide, with a 20-step budget in both arms.

**Architecture:** `a800/alfworld_manifest.py` selects five games per family from the pinned official archive and records game hashes. `a800/evaluate.py` consumes the immutable manifest, records each initial-observation hash and requires the guided hash to match the baseline. `a800/pilot_server.py` starts one server and runs both arms in sequence; results remain in a new private output directory.

**Tech Stack:** Python, unittest/pytest, ALFWorld TextWorld, vLLM, Slurm, one A800.

---

### Task 1: Fixed, balanced selection

**Files:** create `a800/alfworld_manifest.py`, `a800/test_alfworld_manifest.py`.

- [x] Test six families with five selected per family, deterministic order, inclusion of the three previous IDs, unique IDs, and failures for missing anchors or insufficient games.
- [x] Verify tests fail before implementation; implement selection, pinned-archive SHA-256 verification and per-game hashes; verify tests pass.
- [x] Generate one private manifest from the official archive and verify 30 unique games, six families, and three earlier cases.

### Task 2: Pair initial states and request settings

**Files:** modify `a800/baseline_eval.py`, `a800/evaluate.py`, `a800/pilot_server.py`; add tests in their existing test files.

- [x] Test that the baseline records the initial observation hash, the guided arm rejects a different hash before requesting an action, and default prompt behavior is unchanged.
- [x] Test manifest file/hash validation and two-arm command construction; observe the expected failures and implement minimally.
- [x] Verify both arms use identical game order, independent seed 0, static guide only in the guided arm, same generation and server parameters, and preserve completed private episodes.

### Task 3: Verify, run, and report

- [x] Run the full targeted tests locally, copy reviewed code and immutable manifest to a new isolated cluster directory, and run CPU tests there.
- [x] Start one bounded Slurm job, keep user data and old outputs untouched, and monitor both arms. If access or execution fails, report it without inventing a result.
- [x] Compare paired successes by family and initial-hash matches. Record only reviewed code, tests, configuration and aggregate results; check explicit files before a normal Git push.
