"""
GRPO 里能单独手算、单独测试的几块。训练循环在 train_grpo.py，测试在 test_grpo_core.py。

一步 GRPO 按这个顺序走：
  1 采样      同一道题让模型答 G 次，这 G 个答案叫一组
  2 奖励      模拟器给每个答案打分：成功 1，否则 0              reward()
  3 优势      组内比较：比组平均好的为正，比组平均差的为负        group_advantages()
  4 mask      每个答案只算到结束符为止，后面补齐的位置不算        completion_mask()
  5 log 概率  模型给“自己写出的每个 token”的 log 概率          token_logprobs() / policy_logprobs()
  6 loss      优势为正：提高这个答案每个 token 的概率；为负：降低  grpo_loss()
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sft"))
from eval_plans import score  # noqa: E402  和 SFT 评测用同一个判分


def reward(row, text):
    """第一版奖励：动作全部合法、并且最后达成目标 = 1，否则 = 0。返回 (奖励, 判分明细)。
    明细里的 format / legal 只记下来看错误类型，不进奖励。
    注意：8 步成功和 12 步成功都是 1；达成目标后再多做一步无关动作（比如拿起一块不在目标里的积木）也还是 1。"""
    s = score(row, text)
    return (1.0 if s["success"] else 0.0), s


def group_advantages(rewards, eps=1e-4):
    """一组 G 个奖励 → G 个优势：(奖励 − 组平均) / (组标准差 + eps)。
    标准差用总体标准差（除以 G，不是 G−1）；TRL 默认除以 G−1，数值会略有不同。
    全对或全错时每个奖励都等于平均，优势全是 0：这组没有学习信号。eps 只是防止除以 0。"""
    r = torch.tensor(rewards, dtype=torch.float32)
    return (r - r.mean()) / (r.std(unbiased=False) + eps)


def completion_mask(ids, eos_ids):
    """ids: [N, C] 生成出来的答案（右边已补齐）。返回 [N, C]：第一个结束符之前的位置和它自己 = 1，后面 = 0。
    结束符也算：停下来是模型自己选的一个动作。写满 C 个都没停的答案，整行都是 1。"""
    is_eos = torch.isin(ids, torch.tensor(sorted(eos_ids), device=ids.device)).int()
    eos_before = is_eos.cumsum(dim=1) - is_eos  # 当前位置之前（不含自己）出现过几个结束符
    return (eos_before == 0).float()


def token_logprobs(logits, tokens, temperature=1.0):
    """logits: [N, C, V]，已经对齐好：logits[:, j] 是用来预测 tokens[:, j] 的那个输出。tokens: [N, C]。
    返回 [N, C]：每个实际写出的 token 的 log 概率。先除以温度，和采样时用的分布一致。"""
    logp = torch.log_softmax(logits.float() / temperature, dim=-1)
    return logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def policy_logprobs(model, prompt_ids, prompt_mask, comp_ids, temperature=1.0):
    """让模型把“题目 + 它自己写的答案”整段读一遍，取出答案里每个 token 的 log 概率，返回 [N, C]。
    prompt_ids / prompt_mask: [N, P]，左边补齐；comp_ids: [N, C]，右边补齐。
    第 i 个位置的输出预测第 i+1 个 token，所以答案的第 j 个 token 由第 P+j−1 个位置预测：
    只要最后 C+1 个位置的输出（logits_to_keep），再丢掉最后一个（它预测的是答案后面的东西）。"""
    n_comp = comp_ids.shape[1]
    ids = torch.cat([prompt_ids, comp_ids], dim=1)
    attn = torch.cat([prompt_mask, torch.ones_like(comp_ids)], dim=1)
    pos = (attn.cumsum(dim=1) - 1).clamp(min=0)  # 左边补齐时，第一个真 token 的位置记为 0，和 generate 一致
    logits = model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=n_comp + 1,
                   use_cache=False).logits[:, :-1]  # use_cache：只有逐个生成时才需要缓存，这里关掉省内存
    return token_logprobs(logits, comp_ids, temperature)


def grpo_loss(logp, old_logp, adv, mask, n_total, clip_eps=0.2):
    """要最小化的 loss。
    logp: [n, C] 当前模型的 token log 概率（带梯度）；old_logp: [n, C] 采样时的 log 概率（不带梯度）
    adv: [n] 每个答案一个优势，这个答案的每个 token 都用它；mask: [n, C]
    n_total: 这一步一共有几个答案。分几小批算时每批都除以总数，加起来和一次算完相同。

    每个 token：ratio = exp(logp − old_logp) = 新概率 / 旧概率，取 min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)。
    先在每个答案内部对 token 求平均，再对答案求平均（DeepSeekMath GRPO 的写法；
    DAPO 改成所有 token 一起平均，留到 2b 对照）。
    第一版每批只更新一次，old_logp 就是 logp.detach()：ratio 恒等于 1，clip 不会触发，
    梯度等于 −A·∇log 概率 的平均。"""
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(1)
    per_token = torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a)
    per_answer = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return -per_answer.sum() / n_total
