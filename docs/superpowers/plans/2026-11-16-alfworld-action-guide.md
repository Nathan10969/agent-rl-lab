# ALFWorld Action-Guide Comparison Implementation Plan

> **For agentic workers:** execute inline with test-driven development and verification; do not contact other sessions.

**Goal:** Compare the existing three ALFWorld development games with and without one task-independent action-format guide.

**Architecture:** `a800/baseline_eval.py` injects an optional static guide into the original initial message; `a800/evaluate.py` selects the same games and checks the saved baseline; `a800/pilot_server.py` runs the guided ALFWorld-only job with unchanged model settings. Output stays in a new ignored run directory.

**Tech Stack:** Python unittest/pytest, ALFWorld TextWorld, vLLM, Slurm and one A800.

---

### Task 1: Prove the prompt change is isolated

**Files:** modify `a800/test_baseline_eval.py`, `a800/baseline_eval.py`.

- [x] Add a fake environment test that records the first message with default and guided modes; require `guided == original_prefix + static_guide + original_observation`, and require default output to equal the old literal prompt.
- [x] Run `.venv/bin/python -m unittest discover -s a800 -p 'test_baseline_eval.py' -q`; observe the expected missing-guide failure.
- [x] Add `ALFWORLD_ACTION_GUIDE` and an optional `action_guide` argument to `run_alfworld`; never expose per-state admissible actions or change feedback, sampling, rewards or step limit.
- [x] Re-run the test file and verify all tests pass.

### Task 2: Pin the baseline and propagate the flag

**Files:** modify `a800/test_baseline_eval.py`, `a800/test_pilot_server.py`, `a800/alfworld_probe.py`, `a800/evaluate.py`, `a800/pilot_server.py`.

- [x] Add tests that reject a baseline with different task IDs or generation settings and accept the exact saved baseline; add a command-construction test that preserves the default and forwards the guide and baseline directory only to ALFWorld.
- [x] Run the focused tests; verify they fail because reference checking and flags are absent.
- [x] Add explicit CLI flags, validate the baseline before inference, save the guide hash and generate only the ALFWorld commands for this run.
- [x] Run `python -m pytest -q blocksworld/test_blocksworld.py blocksworld/test_step_env.py blocksworld/test_recovery_samples.py a800/test_baseline_eval.py a800/test_pilot_server.py a800/test_build_cases.py`; require zero failures.

### Task 3: Run and report

- [x] Copy the exact reviewed files into a new isolated cluster directory; check file hashes and run the focused CPU tests there.
- [x] Use the previous official model and inference settings in a bounded one-A800 Slurm job; preserve `pilot-02` and write to a new result directory.
- [x] Compare three case IDs, successes, steps and representative action/feedback pairs against the saved baseline. Report the three-case limit and no unsupported causal generalization.
- [x] Record code, configuration and aggregate findings in the learning log; check the selected files for private information and unrelated user-owned changes before the requested normal commit/push.
