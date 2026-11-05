"""续训命令行安全检查：失败时不创建运行目录。"""
import subprocess
import sys
import random
from pathlib import Path

import torch


SCRIPT = Path(__file__).with_name("train_grpo.py")


def test_startup_seeds_python_and_torch_randomness():
    import train_grpo

    assert hasattr(train_grpo, "initialize_randomness")
    train_grpo.initialize_randomness(0)
    expected = (random.random(), torch.rand(3))
    train_grpo.initialize_randomness(0)
    assert random.random() == expected[0]
    assert torch.equal(torch.rand(3), expected[1])


def test_tokenizer_fingerprint_detects_new_template_and_loaded_template_change(tmp_path):
    import train_grpo

    assert hasattr(train_grpo, "tokenizer_fingerprints")
    (tmp_path / "tokenizer.json").write_text("vocab")
    before = train_grpo.tokenizer_fingerprints(tmp_path, "template A")
    (tmp_path / "chat_template.jinja").write_text("new template")
    after_file = train_grpo.tokenizer_fingerprints(tmp_path, "template A")
    assert after_file != before
    assert train_grpo.tokenizer_fingerprints(tmp_path, "template B") != after_file
    templates = tmp_path / "chat_templates"
    templates.mkdir()
    (templates / "tool.jinja").write_text("tool template")
    assert train_grpo.tokenizer_fingerprints(tmp_path, "template A") != after_file


def test_start_adapter_fingerprint_changes_if_config_changes(tmp_path):
    import train_grpo

    assert hasattr(train_grpo, "adapter_fingerprints")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    config = tmp_path / "adapter_config.json"
    config.write_text('{"lora_alpha": 16}')
    first = train_grpo.adapter_fingerprints(tmp_path)
    config.write_text('{"lora_alpha": 32}')
    assert train_grpo.adapter_fingerprints(tmp_path) != first


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def test_rejects_incomplete_resume_before_creating_output(tmp_path):
    old = tmp_path / "old-adapter"
    old.mkdir()
    (old / "adapter_model.safetensors").write_bytes(b"old")
    output = tmp_path / "new-run"
    result = run("--run-dir", output, "--resume-from", old, "--steps", 4)
    assert result.returncode != 0
    assert "完整续训存档" in result.stderr + result.stdout
    assert not output.exists()
    assert (old / "adapter_model.safetensors").read_bytes() == b"old"


def test_rejects_stop_after_later_than_total_without_creating_output(tmp_path):
    output = tmp_path / "new-run"
    result = run("--run-dir", output, "--steps", 2, "--stop-after", 3)
    assert result.returncode != 0
    assert "stop-after" in result.stderr + result.stdout
    assert not output.exists()
