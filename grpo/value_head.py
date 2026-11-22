"""从动作前的 Qwen 前文提取隐藏表示，用独立价值头预测回报。"""
import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        if hidden_size < 1:
            raise ValueError("隐藏表示维度必须大于零")
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden):
        if hidden.ndim != 2 or hidden.shape[1] != self.linear.in_features:
            raise ValueError("隐藏表示必须是 [批大小, hidden_size]")
        return self.linear(hidden.to(dtype=self.linear.weight.dtype)).squeeze(-1)


def extract_prompt_hidden(model, turn):
    """只读取左补齐的动作前前文；不读取动作 token，也不记录策略模型的梯度。"""
    if model.training:
        raise ValueError("提取前请将策略模型设为 eval 模式")
    ids, mask = turn.get("prompt_ids"), turn.get("prompt_mask")
    if not isinstance(ids, (list, tuple)) or not isinstance(mask, (list, tuple)) or not ids or len(ids) != len(mask):
        raise ValueError("前文 token 和 mask 必须是长度相同的非空序列")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("前文 token 必须是非负整数")
    if any(bit not in (0, 1) for bit in mask) or mask[-1] != 1 or any(a > b for a, b in zip(mask, mask[1:])):
        raise ValueError("mask 必须是左侧补齐的 0 后接真实 token 的 1")

    device = next(model.parameters()).device
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_mask = torch.tensor([mask], dtype=torch.long, device=device)
    position_ids = (prompt_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        outputs = model(input_ids=prompt_ids, attention_mask=prompt_mask,
                        position_ids=position_ids, output_hidden_states=True,
                        use_cache=False, logits_to_keep=1, return_dict=True)
        # clone 只保留最后一个位置，不让整段隐藏表示因切片而驻留内存。
        hidden = outputs.hidden_states[-1][:, -1, :].clone()
    return hidden


def predict_trajectory_values(model, value_head, turns):
    """按轨迹先后顺序预测每个动作开始前的价值。"""
    if not turns:
        raise ValueError("轨迹必须至少有一个动作")
    return torch.cat([value_head(extract_prompt_hidden(model, turn)) for turn in turns])


def step_advantages(returns, values):
    """逐步比较回报与价值预测；返回的优势不携带价值头的梯度。"""
    if returns.ndim != 1 or returns.shape != values.shape or returns.numel() == 0:
        raise ValueError("回报与价值预测必须是一一对应的非空一维张量")
    if not torch.isfinite(returns).all() or not torch.isfinite(values).all():
        raise ValueError("回报与价值预测必须是有限值")
    return returns.detach() - values.detach()
