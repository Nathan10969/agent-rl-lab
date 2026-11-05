"""续训存档的 CPU 测试；不加载真实语言模型，也不改已有实验。"""
import importlib.util
import random

import pytest
import torch
from safetensors.torch import load_file, save_file


def test_resume_state_module_is_available():
    assert importlib.util.find_spec("resume_state") is not None


class TinyModel(torch.nn.Linear):
    """真实的小 torch 模型，提供与 PEFT 相同的保存入口。"""
    def __init__(self):
        super().__init__(2, 1, bias=False)

    def save_pretrained(self, path):
        save_file(self.state_dict(), str(path / "adapter_model.safetensors"))
        (path / "adapter_config.json").write_text('{"type":"tiny"}')


def update(model, optimizer, sample):
    (model(sample).sum() ** 2).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_checkpoint_restores_optimizer_order_and_random_stream(tmp_path):
    from resume_state import inspect_checkpoint, restore_training_state, save_checkpoint

    torch.manual_seed(42)
    random.seed(42)
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    update(model, opt, torch.tensor([[1.0, 2.0]]))
    target = tmp_path / "checkpoint-0001"
    save_checkpoint(target, model, opt, 1, [2, 0, 1], ["weight"], {"data": "abc"}, "cpu")
    next_sample = torch.rand(1, 2)
    next_python = random.random()
    update(model, opt, next_sample)

    resumed = TinyModel()
    resumed.load_state_dict(load_file(str(target / "adapter_model.safetensors")))
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    state = inspect_checkpoint(target, {"data": "abc"})
    assert state["completed_step"] == 1
    assert state["order"] == [2, 0, 1]
    restore_training_state(state, resumed_opt, ["weight"], "cpu")
    assert torch.equal(torch.rand(1, 2), next_sample)
    assert random.random() == next_python
    update(resumed, resumed_opt, next_sample)
    assert torch.equal(model.weight, resumed.weight)
    left = opt.state_dict()["state"][0]
    right = resumed_opt.state_dict()["state"][0]
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(left[name], right[name])


def test_checkpoint_rejects_tampering_and_parameter_mismatch(tmp_path):
    from resume_state import inspect_checkpoint, restore_training_state, save_checkpoint

    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    target = tmp_path / "checkpoint-0001"
    save_checkpoint(target, model, opt, 1, [0, 1], ["weight"], {"data": "abc"}, "cpu")
    with pytest.raises(ValueError, match="配置"):
        inspect_checkpoint(target, {"data": "different"})
    state = inspect_checkpoint(target, {"data": "abc"})
    with pytest.raises(ValueError, match="参数"):
        restore_training_state(state, opt, ["different"], "cpu")
    state_file = target / "training_state.pt"
    with state_file.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="指纹"):
        inspect_checkpoint(target, {"data": "abc"})


def test_checkpoint_does_not_overwrite_or_accept_old_adapter(tmp_path):
    from resume_state import inspect_checkpoint, save_checkpoint

    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    old = tmp_path / "old"
    old.mkdir()
    (old / "adapter_model.safetensors").write_bytes(b"old-data")
    with pytest.raises(ValueError, match="完整"):
        inspect_checkpoint(old, {})
    with pytest.raises(FileExistsError):
        save_checkpoint(old, model, opt, 1, [0], ["weight"], {}, "cpu")
    assert (old / "adapter_model.safetensors").read_bytes() == b"old-data"


def test_checkpoint_does_not_publish_invalid_adapter(tmp_path):
    from resume_state import save_checkpoint

    class BrokenAdapter(TinyModel):
        def save_pretrained(self, path):
            (path / "adapter_model.safetensors").write_bytes(b"not safetensors")
            (path / "adapter_config.json").write_text('{"type":"tiny"}')

    model = BrokenAdapter()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    target = tmp_path / "checkpoint-0001"
    with pytest.raises(ValueError, match="adapter"):
        save_checkpoint(target, model, opt, 1, [0], ["weight"], {}, "cpu")
    assert not target.exists()
