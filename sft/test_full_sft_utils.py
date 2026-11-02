"""全量 SFT 的轻量安全测试：无需模型、GPU 或训练数据。"""

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _load_utils():
    path = Path(__file__).with_name("full_sft_utils.py")
    assert path.is_file(), "全量 SFT 工具模块尚未实现"
    spec = importlib.util.spec_from_file_location("full_sft_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_count_includes_partial_final_batch():
    utils = _load_utils()
    assert utils.batch_count(3021, 8) == 378
    assert utils.batch_count(16, 8) == 2
    assert utils.batch_count(0, 8) == 0


@pytest.mark.parametrize("n,batch_size", [(-1, 8), (1, 0), (1, -1)])
def test_batch_count_rejects_invalid_arguments(n, batch_size):
    with pytest.raises(ValueError):
        _load_utils().batch_count(n, batch_size)


def test_create_run_dir_creates_new_nested_directory(tmp_path):
    target = tmp_path / "runs" / "experiment-1"
    _load_utils().create_run_dir(target)
    assert target.is_dir()


def test_create_run_dir_never_overwrites_existing_results(tmp_path):
    target = tmp_path / "experiment-1"
    target.mkdir()
    saved = target / "results.json"
    saved.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _load_utils().create_run_dir(target)
    assert saved.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize(
    "total_steps,interval,expected",
    [(378, 126, [126, 252, 378]), (10, 4, [4, 8, 10]), (3, 10, [3]), (8, 4, [4, 8])],
)
def test_checkpoint_steps_include_last_step_once(total_steps, interval, expected):
    assert _load_utils().checkpoint_steps(total_steps, interval) == expected


@pytest.mark.parametrize("total_steps,interval", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_checkpoint_steps_reject_nonpositive_arguments(total_steps, interval):
    with pytest.raises(ValueError):
        _load_utils().checkpoint_steps(total_steps, interval)


def test_best_checkpoint_uses_lowest_loss_and_earliest_step_for_ties():
    records = [
        {"step": 378, "val_loss": 0.12},
        {"step": 252, "val_loss": 0.10},
        {"step": 126, "val_loss": 0.10},
    ]
    assert _load_utils().pick_best_checkpoint(records) == records[2]


@pytest.mark.parametrize("records", [[], [{"step": 1, "val_loss": float("nan")}],
                                    [{"step": 1, "val_loss": float("inf")}],
                                    [{"step": 1, "val_loss": float("-inf")}]])
def test_best_checkpoint_rejects_empty_or_nonfinite_losses(records):
    with pytest.raises(ValueError):
        _load_utils().pick_best_checkpoint(records)


def test_verify_file_hash_returns_actual_hash_for_matching_file(tmp_path):
    target = tmp_path / "train.jsonl"
    target.write_bytes(b"sample data\n")
    expected = hashlib.sha256(b"sample data\n").hexdigest()
    assert _load_utils().verify_file_hash(target, expected) == expected


def test_verify_file_hash_rejects_mismatch_without_modifying_source(tmp_path):
    target = tmp_path / "train.jsonl"
    content = b"sample data\n"
    target.write_bytes(content)
    incorrect = hashlib.sha256(b"different data\n").hexdigest()
    with pytest.raises(ValueError):
        _load_utils().verify_file_hash(target, incorrect)
    assert target.read_bytes() == content


@pytest.mark.parametrize(
    "tokens,limit,expected",
    [
        ([1, 151645, 151643], 3, {"eos_hit": True, "stop_token_id": 151645, "reached_limit_without_eos": False}),
        ([1, 151643, 151643], 3, {"eos_hit": True, "stop_token_id": 151643, "reached_limit_without_eos": False}),
        ([1, 2, 3], 3, {"eos_hit": False, "stop_token_id": None, "reached_limit_without_eos": True}),
        ([1, 2], 3, {"eos_hit": False, "stop_token_id": None, "reached_limit_without_eos": False}),
    ],
)
def test_generation_stop_identifies_first_eos_and_length_limit(tokens, limit, expected):
    assert _load_utils().generation_stop(tokens, [151645, 151643], limit) == expected
