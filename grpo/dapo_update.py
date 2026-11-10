"""在固定的一篮采样数据上多次更新；旧 log 概率只在更新前计算一次。"""
import math

import torch

from dapo_core import dapo_loss


def optimize_batch(model, optimizer, groups, logprob_fn, updates=2, temperature=1.0,
                   eps_low=0.2, eps_high=0.28, max_grad_norm=1.0):
    if not groups or updates < 1 or temperature <= 0 or max_grad_norm <= 0:
        raise ValueError("必须有题组、正温度、正更新次数和正梯度阈值")
    total_tokens = sum(float(g["mask"].sum().item()) for g in groups)
    if total_tokens <= 0:
        raise ValueError("这一批没有有效 token")
    model.eval()
    # 在任何优化器更新之前缓存旧策略概率；不在后面的循环里刷新。
    with torch.no_grad():
        old = [logprob_fn(model, g["prompt_ids"], g["prompt_mask"], g["comp_ids"], temperature).detach().clone()
               for g in groups]
    parameters = [p for param_group in optimizer.param_groups for p in param_group["params"] if p.requires_grad]
    if not parameters:
        raise ValueError("没有可训练参数")
    history = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        total_loss, ratio_min, ratio_max, clipped, count = 0.0, math.inf, -math.inf, 0, 0
        for g, old_logp in zip(groups, old):
            logp = logprob_fn(model, g["prompt_ids"], g["prompt_mask"], g["comp_ids"], temperature)
            loss = dapo_loss(logp, old_logp, g["advantage"], g["mask"], eps_low, eps_high, total_tokens)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise ValueError("目标不是有限数，停止更新")
            loss.backward()
            total_loss += loss.item()
            with torch.no_grad():
                ratios = torch.exp(logp - old_logp)
                valid = g["mask"].bool()
                if valid.any():
                    ratio_min = min(ratio_min, ratios[valid].min().item())
                    ratio_max = max(ratio_max, ratios[valid].max().item())
                    adv = g["advantage"].unsqueeze(1)
                    clipped += int((((adv > 0) & (ratios > 1 + eps_high)) |
                                    ((adv < 0) & (ratios < 1 - eps_low)))[valid].sum().item())
                    count += int(valid.sum().item())
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        history.append({"loss": total_loss, "grad_norm": float(norm), "ratio_min": ratio_min,
                        "ratio_max": ratio_max, "clip_fraction": clipped / count, "tokens": count})
    optimizer.zero_grad(set_to_none=True)
    return history
