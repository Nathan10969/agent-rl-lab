# GRPO Resume Implementation Plan

> **For agentic workers:** Execute inline in the current session, with test-first checkpoints and a small real MPS comparison. Do not touch existing runs or the held-out evaluation sets.

**Goal:** Save and restore a complete GRPO training-step boundary, then compare uninterrupted and interrupted/resumed runs.

**Architecture:** A small `grpo/resume_state.py` module owns publishing and checking a versioned, integrity-checked checkpoint; `grpo/train_grpo.py` captures/restores model, optimizer, RNG, order and compatibility metadata. Each resumed process writes to a new run directory. Existing adapter-only checkpoints are rejected as full resumes.

**Tech Stack:** Python 3.12, local PyTorch 2.10/PEFT, pytest, MPS; SHA-256 and atomic same-filesystem directory publication.

---

### Task 1: Failing checkpoint and continuation tests

**Files:** Create `grpo/test_resume_state.py`; later create `grpo/resume_state.py`.

- [ ] Test a real CPU `torch.nn.Linear` plus AdamW and CPU RNG. Train one update, save using `save_checkpoint`, perturb Python/Torch state, restore with `load_checkpoint`, and compare next sampled tensor, optimizer state, saved order and next step to an uninterrupted copy. Request `device="cpu"`, `completed_step=1`, `order=[2,0,1]`, `parameter_names=["weight"]`, and compatibility `{"data_sha256":"abc"}`. Assert optimizer step and moment tensors are equal, not merely the model output.
- [ ] Test missing completion marker, edited `training_state.pt`, changed compatibility metadata and changed parameter order each raise an explicit `ValueError` before restoring state. Test an existing final checkpoint is never overwritten.
- [ ] Test the legacy adapter-only directory is rejected and original bytes survive all negative cases.
- [ ] Run `.venv/bin/python -m pytest -q grpo/test_resume_state.py` with pytest installed in the environment; expected red: module/functions missing. Implement the functions only after observing red.

### Task 2: Atomic checkpoint module

**Files:** Create `grpo/resume_state.py`; Test `grpo/test_resume_state.py`.

- [ ] Implement `save_checkpoint(path, model, optimizer, completed_step, order, parameter_names, compatibility, device)` and `load_checkpoint(path, model, optimizer, parameter_names, compatibility, device)`.
- [ ] Save the model by calling `model.save_pretrained` for PEFT models, using a small real torch model adapter in CPU tests. Save a restricted-loadable dictionary containing optimizer state, integer step/order and `torch.get_rng_state()` plus `torch.mps.get_rng_state()` when device is MPS. The final RNG capture occurs after model serialization; resume restores RNG only after all model/optimizer reconstruction.
- [ ] Write files into a newly allocated sibling partial directory; hash the model/config and state, write versioned JSON manifest last, then publish by renaming after checking the final path is absent. Never overwrite an existing final checkpoint or delete pre-existing data. Validate required files, checksums, metadata, parameter names/order and format before loading. Use `torch.load(..., map_location="cpu", weights_only=True)` only for trusted local checkpoints.
- [ ] Run checkpoint tests green. Add a test if a discovered edge case was not already covered; rerun before continuing.

### Task 3: Connect the existing training loop

**Files:** Modify `grpo/train_grpo.py`; create `grpo/test_train_resume.py`.

- [ ] First test CLI rejects invalid resume requests and invalid stop step, refuses existing output directories, and keeps `--steps` as the total target. Expected red before changes.
- [ ] Add `--resume-from` and `--stop-after`; choose the source adapter from a validated checkpoint when resuming, otherwise the existing SFT adapter. Save full states at each `--save-every` boundary and an intentional stop. Keep sampling/GRPO settings unchanged. Restore optimizer after recreating the exact named trainable parameter order; restore saved order and RNG last; iterate from `completed_step + 1` to the requested total.
- [ ] Include stable compatibility fields: model/start-adapter/train-data/tokenizer/code fingerprints, torch/transformers/peft/Python versions, device, precision, total steps, per-step prompt count, group size, sampling, clipping and optimizer settings. Refuse incompatible inputs before sampling. Keep data order and source path provenance, and never rewrite old run records.
- [ ] Persist actual effective generated token IDs per answer in the new run's sample log, alongside answer, reward, advantage and question ID; do not change reward computation. Summaries from resumed processes distinguish segment steps and cumulative target; avoid treating segment totals as all-run totals.
- [ ] Run all unit and CLI tests. Check old GRPO tests remain green.

### Task 4: Real MPS comparison

**Files:** Run `grpo/train_grpo.py`; create a small comparison utility/test as needed after a failing test; store machine-readable results under a fresh ignored `outputs/grpo/` directory.

- [ ] Run A with total 4, save interval 2; run B with identical settings and deliberate stop after step 2; start a separate process from B's step-2 checkpoint with target 4. Never modify the prior `trial-01`.
- [ ] Compare IDs, effective token IDs, rewards and advantages of steps 3–4 and compare all LoRA and AdamW tensors, integer state and next RNG states at step 4. Report exact identity and maximum absolute floating-point difference separately; any mismatch is a result, not a silent pass.
- [ ] Run `.venv/bin/python -m pytest -q blocksworld/test_blocksworld.py sft/test_full_sft_utils.py sft/test_train_full.py grpo/test_grpo_core.py grpo/test_resume_state.py grpo/test_train_resume.py`; report the real output. Update a concise learning record and README only after results are known.

### Task 5: Handoff

- [ ] Run `git diff --check`, inspect changed files, confirm old outputs/checkpoints are untouched, and report local modifications; do not push or commit unless explicitly requested. Explain saving and restoring line by line with the actual comparison result and its same-machine limitations.
