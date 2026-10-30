# SFT Memory Learning Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the first SFT trial and memory investigation accurately, and publish the reviewed learning scripts.

**Architecture:** Update the repository's README with a concise status and link to a standalone lesson. The lesson summarizes the existing local evidence without moving generated artifacts into Git or modifying another collaborator's training scripts.

**Tech Stack:** Markdown, Python, PyTorch MPS, Git.

---

### Task 1: Verify the evidence

**Files:**
- Read: `outputs/sft/qwen0.5b-lora-trial-run1/results.json`
- Read: `outputs/sft/qwen0.5b-lora-trial/results.json`
- Read: `outputs/sft/mem_probe.log`

- [ ] Compare the two runs' training duration, validation loss and scores using their JSON fields; inspect the probe's final driver readings.
- [ ] Run `shasum -a 256 outputs/sft/qwen0.5b-lora-trial-run1/results.json outputs/sft/qwen0.5b-lora-trial/results.json outputs/sft/mem_probe.log`; record the resulting fingerprints and note that these ignored files remain local.

### Task 2: Write the learning record

**Files:**
- Modify: `README.md`
- Create: `docs/lessons/02-first-sft-and-mps-memory.md`

- [ ] Update README status and add a link to the new lesson; warn that running the existing fixed-output training script can overwrite local artifacts.
- [ ] Write the lesson in the sequence observed problem, measured comparison, implemented changes, training results, limitations and next learning step. Include the real experiment settings and file fingerprints; distinguish system swap, process footprint and per-step MPS driver allocation.
- [ ] Explicitly state that the evidence does not establish the sole root cause, absence of leaks, full-run memory peak or system-wide zero swap.

### Task 3: Verify and publish

**Files:**
- Review: `sft/data.py`, `sft/eval_plans.py`, `sft/train_lora.py`, `sft/mem_probe.py`, `sft/requirements.txt`

- [ ] Run `python -m pytest -q blocksworld/test_blocksworld.py`; expect all 17 existing tests to pass. Do not run training or overwrite existing results.
- [ ] Parse the four SFT Python files with `ast.parse`, inspect `git diff --check`, and report that these checks do not constitute an SFT training test.
- [ ] Stage only the planned README, lesson, plan and five SFT files; inspect staged changes, commit under Jony Wong and push the current branch after verifying the remote. Do not stage model weights, data or ignored outputs.
