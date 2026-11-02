"""grpo_core.py 的测试。数字都能手算；最后一个测试用一个很小的随机 Qwen2，在 CPU 上几秒跑完，不加载真模型。
运行：.venv/bin/python -m pytest grpo/test_grpo_core.py -q"""
import math

import pytest
import torch

from grpo_core import completion_mask, group_advantages, grpo_loss, policy_logprobs, reward, token_logprobs

# ---------- 奖励：用一道手写的 4 块部分目标题，不读 data/ ----------
# 初始：C 在桌上，B 在 C 上，D 在 B 上，A 在 D 上。目标：A 在 C 上，C 在 D 上（B 放哪都行）
ROW = {"init": [["C", "B", "D", "A"]], "goal": [["A", "C"], ["C", "D"]]}
GOOD = "unstack(A, D)\nputdown(A)\nunstack(D, B)\nputdown(D)\nunstack(B, C)\nputdown(B)\npickup(C)\nstack(C, D)\npickup(A)\nstack(A, C)"


def test_reward_success_is_one():
    r, s = reward(ROW, GOOD)
    assert r == 1.0 and s == {"format": True, "legal": True, "success": True}


@pytest.mark.parametrize("text, detail", [
    ("unstack(A, D)\nstack(C,桌上)", {"format": False, "legal": False, "success": False}),  # 采样里见过的格式错误
    ("pickup(A)", {"format": True, "legal": False, "success": False}),  # A 在 D 上，不在桌上，非法
    ("unstack(A, D)\nputdown(A)", {"format": True, "legal": True, "success": False}),  # 合法但没达成目标
])
def test_reward_failures_are_zero_and_keep_error_type(text, detail):
    r, s = reward(ROW, text)
    assert r == 0.0 and s == detail


def test_reward_ignores_extra_steps_after_goal():
    """第一版说好的语义：达成目标后再拿起 B（目标没提到 B），仍然算成功，手里拿着东西也不扣分。"""
    r, _ = reward(ROW, GOOD + "\npickup(B)")
    assert r == 1.0


# ---------- 组内优势 ----------

def test_advantages_three_of_eight():
    """8 个里成功 3 个：平均 0.375，总体标准差 √(0.375×0.625) ≈ 0.4841。
    成功：0.625 / 0.4841 ≈ +1.291；失败：−0.375 / 0.4841 ≈ −0.775。一组加起来是 0。"""
    adv = group_advantages([1, 0, 1, 0, 0, 1, 0, 0])
    assert adv[0].item() == pytest.approx(0.625 / math.sqrt(0.375 * 0.625), abs=1e-3)
    assert adv[1].item() == pytest.approx(-0.375 / math.sqrt(0.375 * 0.625), abs=1e-3)
    assert adv.sum().item() == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("rewards", [[1] * 8, [0] * 8])
def test_all_same_group_has_zero_advantage(rewards):
    assert torch.equal(group_advantages(rewards), torch.zeros(8))


# ---------- mask ----------

def test_completion_mask_stops_after_first_eos():
    eos, pad = 3, 0  # 真模型里 <|im_end|> 和补齐用的 <|endoftext|> 都是结束符
    ids = torch.tensor([[5, 7, eos, pad, pad],  # 写了 2 个动作 token 再停：前 3 个算
                        [5, 7, 8, 9, 10],  # 写满还没停：全算
                        [eos, pad, pad, pad, pad]])  # 一上来就停：只有结束符算
    assert completion_mask(ids, {eos, pad}).tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]


# ---------- token log 概率 ----------

def test_token_logprobs_by_hand():
    """两个候选 token，logits = [0, ln 3]，概率 = [1/4, 3/4]。温度 2 时 logits 变成 [0, ln√3]，概率 = √3/(1+√3)。"""
    logits = torch.tensor([[[0.0, math.log(3)], [0.0, math.log(3)]]])
    assert token_logprobs(logits, torch.tensor([[1, 0]]))[0].tolist() == pytest.approx([math.log(3 / 4), math.log(1 / 4)])
    t2 = token_logprobs(logits, torch.tensor([[1, 1]]), temperature=2.0)
    assert t2[0, 0].item() == pytest.approx(math.log(math.sqrt(3) / (1 + math.sqrt(3))))


# ---------- loss ----------

def test_loss_with_ratio_one_is_zero_but_gradient_is_not():
    """old = logp.detach() 时 ratio = 1。优势 [+1, −1] 加起来是 0，所以 loss 的数值是 0，但梯度不是 0：
    答案 0 有 2 个有效 token，每个 token 的梯度 = −A / (2 × 2 个答案) = −0.25（loss 往下走 → 这个 token 的 log 概率往上）
    答案 1 有 3 个有效 token，每个 = +1 / (3 × 2) ≈ +0.1667；被 mask 的位置梯度是 0。"""
    logp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss = grpo_loss(logp, logp.detach(), torch.tensor([1.0, -1.0]), mask, n_total=2)
    loss.backward()
    assert loss.item() == pytest.approx(0.0)
    assert logp.grad.tolist() == [pytest.approx([-0.25, -0.25, 0.0]), pytest.approx([1 / 6, 1 / 6, 1 / 6])]


@pytest.mark.parametrize("ratio, adv, loss_value, grad", [
    (1.5, +1.0, -1.2, 0.0),  # 好答案的概率已经涨了 50%，超过 1+ε：取裁剪后的 1.2，不再往上推
    (1.5, -1.0, 1.5, 1.5),  # 坏答案概率涨了：min 取没裁剪的 −1.5，继续往下压
    (0.5, +1.0, -0.5, -0.5),  # 好答案概率掉了一半：没裁剪，继续往上推
])
def test_clip_by_hand(ratio, adv, loss_value, grad):
    old = torch.tensor([[-1.0]])
    logp = (old + math.log(ratio)).requires_grad_()
    loss = grpo_loss(logp, old, torch.tensor([adv]), torch.ones(1, 1), n_total=1, clip_eps=0.2)
    loss.backward()
    assert loss.item() == pytest.approx(loss_value)
    assert logp.grad.item() == pytest.approx(grad)


def test_microbatches_add_up_to_full_batch():
    torch.manual_seed(0)
    logp, old = torch.randn(4, 5), torch.randn(4, 5)
    adv, mask = torch.randn(4), (torch.rand(4, 5) > 0.3).float()
    full = grpo_loss(logp, old, adv, mask, n_total=4)
    parts = grpo_loss(logp[:2], old[:2], adv[:2], mask[:2], n_total=4) + grpo_loss(logp[2:], old[2:], adv[2:], mask[2:], n_total=4)
    assert parts.item() == pytest.approx(full.item(), abs=1e-6)


# ---------- token 对齐：采样时的 log 概率 = 训练时重算的 log 概率 ----------

def test_policy_logprobs_match_sampling_and_only_lora_gets_gradient():
    """如果答案的 token 和预测它的位置错开一格，两边的 log 概率会差很多；对齐了只差浮点误差。"""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    pad, eos = 0, 3
    cfg = Qwen2Config(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      initializer_range=0.5, pad_token_id=pad, eos_token_id=[eos, pad])
    model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                                             target_modules=["q_proj", "v_proj"],
                                                             init_lora_weights=False))
    model.eval()
    # 两道题，长度 3 和 5，左边补齐到 5；每道题采 2 个答案，和真训练一样按题排好
    prompt_ids = torch.tensor([[pad, pad, 11, 12, 13], [21, 22, 23, 24, 25]]).repeat_interleave(2, 0)
    prompt_mask = (torch.arange(5) >= torch.tensor([[2], [0]])).long().repeat_interleave(2, 0)
    n_new = 6
    with torch.no_grad():
        out = model.generate(input_ids=prompt_ids, attention_mask=prompt_mask, do_sample=True, temperature=1.0,
                             top_p=1.0, top_k=0, max_new_tokens=n_new, pad_token_id=pad, eos_token_id=[eos, pad],
                             return_dict_in_generate=True, output_logits=True)
    comp = out.sequences[:, 5:]
    comp = torch.nn.functional.pad(comp, (0, n_new - comp.shape[1]), value=pad)
    sampled = token_logprobs(torch.stack(out.logits, dim=1), out.sequences[:, 5:])
    mask = completion_mask(comp, {eos, pad})[:, :sampled.shape[1]]

    logp = policy_logprobs(model, prompt_ids, prompt_mask, comp)
    diff = ((logp[:, :sampled.shape[1]] - sampled).abs() * mask).max().item()
    assert diff < 1e-4, f"采样时和重算的 log 概率差 {diff}，token 很可能没对齐"

    grpo_loss(logp, logp.detach(), torch.tensor([1.0, -1.0, 1.0, -1.0]), completion_mask(comp, {eos, pad}), 4).backward()
    with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
    assert with_grad and all("lora_" in n for n in with_grad)
