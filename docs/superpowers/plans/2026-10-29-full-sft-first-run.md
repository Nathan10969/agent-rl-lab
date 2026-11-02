# First Full Blocksworld SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one full-data SFT epoch safely, save checkpoints and evaluate the selected adapter.

**Architecture:** Keep the previous SFT trial untouched. Add a separate full-run entry point and pure helper functions for output safety, step counts and checkpoint selection; each run writes only to a new ignored output directory.

**Tech Stack:** Python 3.12, PyTorch MPS, Transformers, PEFT, pytest.

---

### Task 1: Test and implement safe run helpers

**Files:** Create `sft/test_full_sft_utils.py` and `sft/full_sft_utils.py`.

- [ ] Test that 3021 examples with batch 8 gives 378 steps, 378/126 produces `[126, 252, 378]`, and 5 items remain in the final batch.
- [ ] Test that creating an existing run directory fails without altering existing files.
- [ ] Test lowest finite validation-loss selection with deterministic earliest-step tie breaking; reject empty and nonfinite records.
- [ ] Observe the tests fail before implementing the four standard-library helper functions; run them again to verify success.

### Task 2: Implement and verify the independent runner

**Files:** Create `sft/train_full.py`; leave `sft/train_lora.py` unchanged.

- [ ] Use the existing dataset, tokenizer, model, LoRA and simulator functions; build a new run directory that never overwrites a pre-existing directory.
- [ ] Record model, code and data hashes, versions, training/validation settings and dataset counts before training. Reject lengths above 320 and incorrect split counts.
- [ ] Evaluate the base policy on all 378 validation tasks; train 378 steps and record loss plus sampled MPS allocations. On steps 126, 252 and 378 save an adapter and token-weighted validation loss, then evaluate the lowest-loss checkpoint on the same validation set.
- [ ] Implement a separately named smoke mode with a maximum of two training batches and two validation examples; no checkpoint or adapter from it may be confused with the full run. Verify it does not overwrite an existing run directory.
- [ ] Run `python -m pytest -q sft/test_full_sft_utils.py blocksworld/test_blocksworld.py` and a short MPS smoke run in a newly generated independent output directory; check results and saved adapter without touching prior trial output.

### Task 3: Launch and report

**Files:** Use ignored `outputs/sft/<new-run>/` for data and artifacts; do not commit model weights.

- [ ] Check physical memory pressure, disk space and that no competing SFT run is active.
- [ ] Launch `sft/train_full.py` unbuffered with the confirmed full configuration and a new path. Capture a persistent log, the PID, and progress through initial loading.
- [ ] Report actual log/output paths, PID, validated settings, current stage and the fact that completion is pending. Do not claim expected success or duration as measured outcomes.
