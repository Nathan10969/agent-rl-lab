"""用一个两 token 小模型检查 DAPO 的多次更新，不加载大模型。"""
import pytest
import torch

from dapo_update import optimize_batch


class TwoTokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2))


def log_probs(model, prompt_ids, prompt_mask, comp_ids, temperature):
    logp = torch.log_softmax(model.logits / temperature, dim=0)
    return logp[comp_ids]


def group():
    return {
        "prompt_ids": torch.ones(2, 1, dtype=torch.long),
        "prompt_mask": torch.ones(2, 1, dtype=torch.long),
        "comp_ids": torch.tensor([[0, 0, 0], [1, 1, 1]]),
        "mask": torch.tensor([[1., 0., 0.], [1., 1., 1.]]),
        "advantage": torch.tensor([1., -1.]),
    }


def test_two_updates_reuse_frozen_old_probabilities():
    model = TwoTokenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    stats = optimize_batch(model, optimizer, [group()], log_probs, updates=2, max_grad_norm=10.0)
    assert len(stats) == 2
    assert stats[0]["ratio_min"] == pytest.approx(1.0)
    assert stats[0]["ratio_max"] == pytest.approx(1.0)
    assert stats[1]["ratio_min"] < 1.0 < stats[1]["ratio_max"]
    assert model.logits[0].item() > 0 > model.logits[1].item()


def test_no_effective_tokens_fails_before_update():
    model = TwoTokenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    sample = group()
    sample["mask"].zero_()
    with pytest.raises(ValueError):
        optimize_batch(model, optimizer, [sample], log_probs, updates=2)
    assert torch.equal(model.logits, torch.zeros(2))


def test_optimizer_excludes_frozen_model_parameters():
    model = TwoTokenModel()
    model.frozen = torch.nn.Parameter(torch.tensor([9.0]), requires_grad=False)
    optimizer = torch.optim.SGD([model.logits], lr=0.2)
    optimize_batch(model, optimizer, [group()], log_probs, updates=2)
    assert model.frozen.item() == 9.0
