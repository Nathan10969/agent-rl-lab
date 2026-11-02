"""全量训练入口的轻量行为测试：不加载模型，不启动训练。"""
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("train_full.py")


def test_runner_offers_help_without_starting_training():
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--run-dir" in result.stdout
    assert "--smoke" in result.stdout


def test_existing_output_directory_is_rejected_without_overwriting(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    marker = run_dir / "results.json"
    marker.write_text("old results")
    result = subprocess.run([sys.executable, str(RUNNER), "--run-dir", str(run_dir), "--smoke"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert marker.read_text() == "old results"
    assert "已存在" in result.stderr
