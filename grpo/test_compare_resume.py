"""对照报告必须检查 token、参数和优化器，而不是只看成功数。"""
import json

import pytest
import torch

from resume_state import save_checkpoint
from test_resume_state import TinyModel, update


def make_run(path, model, opt, token, group_size=1):
    path.mkdir()
    sample = {"step": 3, "id": "q1", "k": 0, "token_ids": token,
              "reward": 1.0, "advantage": 0.5, "text": "ok"}
    rows = [{**sample, "step": step, "k": k} for step in (3, 4) for k in range(group_size)]
    (path / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    compatibility = {"prompts_per_step": 1, "group_size": group_size}
    (path / "config.json").write_text(json.dumps({"resume_from": None}))
    save_checkpoint(path / "checkpoint-0002", model, opt, 2, [0, 1], ["weight"], compatibility, "cpu")
    save_checkpoint(path / "checkpoint-0004", model, opt, 4, [0, 1], ["weight"], compatibility, "cpu")


def mark_resumed(run, source):
    (run / "config.json").write_text(json.dumps({"resume_from": str(source)}))


def test_comparison_catches_changed_token_even_when_text_and_reward_match(tmp_path):
    from compare_resume import compare_runs

    torch.manual_seed(42)
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    update(model, opt, torch.tensor([[1.0, 2.0]]))
    a, b = tmp_path / "a", tmp_path / "b"
    make_run(a, model, opt, [1, 2, 3])
    make_run(b, model, opt, [1, 2, 3])
    mark_resumed(b, a / "checkpoint-0002")
    report = compare_runs(a, b, 3, 4)
    assert report["tokens_equal"] is True
    assert report["adapter_equal"] is True
    assert report["optimizer_equal"] is True
    rows = [json.loads(line) for line in (b / "samples.jsonl").read_text().splitlines()]
    rows[0]["token_ids"] = [1, 2, 4]
    (b / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert compare_runs(a, b, 3, 4)["tokens_equal"] is False


def test_comparison_rejects_identically_truncated_logs(tmp_path):
    from compare_resume import compare_runs

    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    a, b = tmp_path / "a", tmp_path / "b"
    for run in (a, b):
        make_run(run, model, opt, [1, 2], group_size=2)
        rows = [json.loads(line) for line in (run / "samples.jsonl").read_text().splitlines()]
        rows = [row for row in rows if not (row["step"] == 3 and row["k"] == 1)]
        (run / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    mark_resumed(b, a / "checkpoint-0002")
    with pytest.raises(ValueError, match="数量"):
        compare_runs(a, b, 3, 4)


def test_comparison_rejects_self_comparison_or_non_resume(tmp_path):
    from compare_resume import compare_runs

    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    a, b = tmp_path / "a", tmp_path / "b"
    make_run(a, model, opt, [1, 2])
    make_run(b, model, opt, [1, 2])
    with pytest.raises(ValueError, match="同一"):
        compare_runs(a, a, 3, 4)
    with pytest.raises(ValueError, match="续训来源"):
        compare_runs(a, b, 3, 4)


def test_comparison_detects_adapter_config_difference(tmp_path):
    from compare_resume import compare_runs

    class ScaledModel(TinyModel):
        def save_pretrained(self, path):
            super().save_pretrained(path)
            (path / "adapter_config.json").write_text('{"type":"tiny","lora_alpha":32}')

    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    changed = ScaledModel()
    changed.load_state_dict(model.state_dict())
    changed_opt = torch.optim.AdamW(changed.parameters(), lr=0.01)
    a, b = tmp_path / "a", tmp_path / "b"
    make_run(a, model, opt, [1, 2])
    make_run(b, changed, changed_opt, [1, 2])
    mark_resumed(b, a / "checkpoint-0002")
    result = compare_runs(a, b, 3, 4)
    assert result["adapter_equal"] is True
    assert result["adapter_config_equal"] is False


def test_adapter_config_normalizes_unordered_target_modules(tmp_path):
    from compare_resume import adapter_config

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "adapter_config.json").write_text('{"target_modules":["q_proj","v_proj"]}')
    (b / "adapter_config.json").write_text('{"target_modules":["v_proj","q_proj"]}')
    assert adapter_config(a) == adapter_config(b)
