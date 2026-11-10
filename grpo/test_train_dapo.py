"""训练入口的题组组装和保护行为：人工数据，不下载模型。"""
import pytest
import subprocess
import sys
import torch
from pathlib import Path

from train_dapo import check_sampling_logp, prepare_group, save_adapter_step


def test_prepare_group_separates_success_shaping_and_truncation_mask():
    comp = torch.tensor([[5, 3, 0, 0], [6, 7, 8, 9]])
    prompt = torch.tensor([[11, 12], [11, 12]])

    def score(row, text):
        return (1.0 if text == "good" else 0.0), {"success": text == "good"}

    group = prepare_group({"id": "q1"}, prompt, torch.ones_like(prompt), comp,
                          ["good", "unfinished"], {0, 3}, score, max_length=4, buffer=2)
    assert group["success"] == [1, 0]
    assert group["lengths"] == [2, 4]
    assert group["truncated"] == [False, True]
    assert group["shaped_rewards"] == pytest.approx([1.0, -1.0])
    assert group["mask"].tolist() == [[1, 1, 0, 0], [0, 0, 0, 0]]
    assert group["advantage"][0].item() > 0 > group["advantage"][1].item()


def test_prepare_group_rejects_inconsistent_batch_size():
    with pytest.raises(ValueError):
        prepare_group({"id": "q"}, torch.ones(2, 1), torch.ones(2, 1), torch.ones(2, 4),
                      ["one"], {0}, lambda row, text: (0, {}), max_length=4, buffer=2)


def test_each_completed_step_saves_without_overwriting(tmp_path):
    class TinyAdapter:
        def save_pretrained(self, path):
            (path / "adapter.txt").write_text("first result")

    save_adapter_step(TinyAdapter(), tmp_path, 1)
    checkpoint = tmp_path / "checkpoint-0001" / "adapter.txt"
    assert checkpoint.read_text() == "first result"
    with pytest.raises(FileExistsError):
        save_adapter_step(TinyAdapter(), tmp_path, 1)
    assert checkpoint.read_text() == "first result"


def test_sampling_logp_check_ignores_padding_but_rejects_mismatch():
    sampled = torch.tensor([[-1., -2., 99.]])
    mask = torch.tensor([[1, 1, 0]])
    assert check_sampling_logp(sampled, torch.tensor([[-1., -2., 0.]]), mask) == 0
    with pytest.raises(ValueError, match="采样"):
        check_sampling_logp(sampled, torch.tensor([[-2., -1., 0.]]), mask)


def test_existing_output_directory_is_rejected_before_training(tmp_path):
    script = Path(__file__).with_name("train_dapo.py")
    result = subprocess.run([sys.executable, str(script), "--run-dir", str(tmp_path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "拒绝覆盖" in result.stderr


def test_simulator_to_tiny_qwen_lora_updates_end_to_end():
    """真实模拟器 + 极小随机 Qwen + LoRA；答案是测试构造，不声称模型生成了它们。"""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from dapo_core import collect_groups
    from dapo_update import optimize_batch
    from grpo_core import policy_logprobs, reward

    torch.manual_seed(2)
    cfg = Qwen2Config(vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
                      pad_token_id=0, eos_token_id=3)
    model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                                            target_modules=["q_proj", "v_proj"]))
    model.eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=0.01, weight_decay=0)
    row = {"id": "manual", "init": [["A"], ["B"]], "goal": [["A", "B"]]}
    prompt = torch.tensor([[10, 11], [10, 11]])
    comp = torch.tensor([[5, 3, 0, 0], [6, 3, 0, 0]])
    group = prepare_group(row, prompt, torch.ones_like(prompt), comp,
                          ["pickup(A)\nstack(A, B)", "pickup(B)\nstack(B, A)"],
                          {0, 3}, reward, max_length=4, buffer=1)
    assert group["success"] == [1, 0]
    selected, stats = collect_groups(iter([group]), target=1, max_attempts=1)
    history = optimize_batch(model, optimizer, selected, policy_logprobs, updates=2)
    assert stats["mixed"] == 1 and len(history) == 2
    assert history[0]["ratio_min"] == pytest.approx(1.0, abs=1e-6)
    assert history[1]["ratio_min"] != pytest.approx(1.0, abs=1e-6)
    assert any(not torch.equal(a, b) for a, b in zip(before, trainable))
