"""DAPO 基础机制的手算测试：不加载模型，也不启动训练。"""
import math

import pytest
import torch

from dapo_core import collect_groups, dapo_loss, soft_length_penalty


@pytest.mark.parametrize("length, expected", [(100, 0.0), (140, 0.0), (150, -0.5), (160, -1.0), (170, -1.0)])
def test_length_penalty(length, expected):
    assert soft_length_penalty(length, max_length=160, buffer=20) == pytest.approx(expected)


@pytest.mark.parametrize("maximum, buffer", [(0, 1), (160, 0), (160, 161)])
def test_invalid_length_config(maximum, buffer):
    with pytest.raises(ValueError):
        soft_length_penalty(10, max_length=maximum, buffer=buffer)


def test_dynamic_sampling_skips_homogeneous_groups_and_fills():
    candidates = iter([
        {"id": "A", "success": [1, 0]}, {"id": "B", "success": [0, 0]},
        {"id": "C", "success": [1, 1]}, {"id": "D", "success": [0, 1]},
        {"id": "E", "success": [1, 0]}, {"id": "F", "success": [0, 1]},
    ])
    selected, stats = collect_groups(candidates, target=4, max_attempts=10)
    assert [x["id"] for x in selected] == ["A", "D", "E", "F"]
    assert stats == {"attempts": 6, "mixed": 4, "all_wrong": 1, "all_correct": 1, "no_train_signal": 0}


def test_dynamic_sampling_stops_at_budget_without_partial_batch():
    candidates = iter([{"id": str(i), "success": [0, 0]} for i in range(10)])
    selected, stats = collect_groups(candidates, target=4, max_attempts=3)
    assert selected == []
    assert stats["attempts"] == 3 and stats["all_wrong"] == 3


def test_invalid_success_values_rejected():
    with pytest.raises(ValueError):
        collect_groups(iter([{"success": [0, 0.5]}]), target=1, max_attempts=1)


def test_mixed_group_without_trainable_signal_is_replaced():
    no_tokens = {"id": "truncated", "success": [1, 0], "mask": torch.zeros(2, 2),
                 "advantage": torch.tensor([1., -1.])}
    zero_adv = {"id": "same_reward", "success": [1, 0], "mask": torch.ones(2, 2),
                "advantage": torch.zeros(2)}
    usable = {"id": "usable", "success": [1, 0], "mask": torch.ones(2, 2),
              "advantage": torch.tensor([1., -1.])}
    selected, stats = collect_groups(iter([no_tokens, zero_adv, usable]), target=1, max_attempts=3)
    assert selected == [usable]
    assert stats["attempts"] == 3 and stats["no_train_signal"] == 2


@pytest.mark.parametrize("ratio, advantage, expected_loss, expected_grad", [
    (1.25, 1.0, -1.25, -1.25),  # 1.25 未超过 DAPO 上界 1.28
    (1.30, 1.0, -1.28, 0.0),    # 超过上界，正优势停止继续推高
    (0.70, -1.0, 0.80, 0.0),   # 低于下界，负优势停止继续压低
    (1.30, -1.0, 1.30, 1.30),  # 负优势向错误方向移动时仍有梯度
])
def test_asymmetric_clip_and_gradient(ratio, advantage, expected_loss, expected_grad):
    old = torch.tensor([[-1.0]])
    logp = (old + math.log(ratio)).requires_grad_()
    loss = dapo_loss(logp, old, torch.tensor([advantage]), torch.ones(1, 1))
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss, abs=1e-6)
    assert logp.grad.item() == pytest.approx(expected_grad, abs=1e-6)


def test_token_mean_and_mask():
    logp = torch.tensor([[-1., -1., -1., -1.], [-1., -1., -1., -1.]], requires_grad=True)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.]])
    loss = dapo_loss(logp, logp.detach(), torch.tensor([1., -1.]), mask)
    loss.backward()
    assert loss.item() == pytest.approx(1 / 3)
    assert logp.grad[0].tolist() == pytest.approx([-1 / 6, -1 / 6, 0, 0])
    assert logp.grad[1].tolist() == pytest.approx([1 / 6] * 4)


def test_microbatches_share_global_denominator():
    logp = torch.tensor([[-1., -2., -3.], [-1., -2., -3.]], requires_grad=True)
    old = logp.detach().clone()
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    advantage = torch.tensor([1., -1.])
    full = dapo_loss(logp, old, advantage, mask)
    split = dapo_loss(logp[:1], old[:1], advantage[:1], mask[:1], total_tokens=4)
    split += dapo_loss(logp[1:], old[1:], advantage[1:], mask[1:], total_tokens=4)
    assert full.item() == pytest.approx(split.item())


def test_empty_mask_rejected():
    with pytest.raises(ValueError):
        dapo_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 2))
