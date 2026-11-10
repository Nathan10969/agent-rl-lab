"""DAPO 的小组件：动态采样、长度惩罚和非对称 token-level 目标。"""
from itertools import islice

import torch


def soft_length_penalty(length, max_length, buffer):
    """论文的线性长度惩罚；max_length 是生成上限，buffer 是末尾惩罚区间。"""
    if not (isinstance(length, int) and length >= 0 and isinstance(max_length, int)
            and isinstance(buffer, int) and max_length > 0 and 0 < buffer <= max_length):
        raise ValueError("长度必须非负，且 0 < buffer <= max_length")
    return max(-1.0, min(0.0, (max_length - buffer - length) / buffer))


def collect_groups(candidates, target, max_attempts):
    """依次检查候选题组；仅保留原始 0/1 成功标记有差异的组。

    候选由调用方在固定旧策略下实际生成；若预算用尽仍不足，返回空篮子，不能做不完整更新。
    """
    if target < 1 or max_attempts < 1:
        raise ValueError("target 和 max_attempts 必须大于 0")
    stats = {"attempts": 0, "mixed": 0, "all_wrong": 0, "all_correct": 0, "no_train_signal": 0}
    selected = []
    for group in islice(candidates, max_attempts):
        success = group["success"]
        if len(success) < 2 or any(x not in (0, 1) for x in success):
            raise ValueError("每组需要至少两个 0/1 成功标记")
        stats["attempts"] += 1
        if all(x == 0 for x in success):
            stats["all_wrong"] += 1
        elif all(x == 1 for x in success):
            stats["all_correct"] += 1
        else:
            if "mask" in group and "advantage" in group:
                mask, advantage = group["mask"], group["advantage"]
                if mask.ndim != 2 or advantage.shape != (mask.shape[0],) or mask.shape[0] != len(success):
                    raise ValueError("题组 mask 或优势形状不匹配")
                if not (mask.bool() & (advantage.unsqueeze(1) != 0)).any():
                    stats["no_train_signal"] += 1
                    continue
            stats["mixed"] += 1
            selected.append(group)
            if len(selected) == target:
                return selected, stats
    return [], stats


def dapo_loss(logp, old_logp, advantage, mask, eps_low=0.2, eps_high=0.28, total_tokens=None):
    """对所有有效 token 平均裁剪目标；分批计算时传入整批有效 token 总数。"""
    if logp.ndim != 2 or old_logp.shape != logp.shape or mask.shape != logp.shape:
        raise ValueError("logp、old_logp 和 mask 必须是相同的二维形状")
    if advantage.shape != (logp.shape[0],) or not (0 <= eps_low < 1 and eps_high >= 0):
        raise ValueError("优势形状或裁剪阈值无效")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("mask 只能包含 0 和 1")
    denominator = mask.sum().item() if total_tokens is None else total_tokens
    if denominator <= 0:
        raise ValueError("有效 token 数必须大于 0")
    ratio = torch.exp(logp - old_logp.detach())
    a = advantage.detach().unsqueeze(1)
    raw = ratio * a
    clipped = ratio.clamp(1 - eps_low, 1 + eps_high) * a
    return -(torch.minimum(raw, clipped) * mask).sum() / denominator
