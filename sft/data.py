"""
SFT 数据准备：把一条样本变成训练用的 input_ids 和 labels，并打印出来人工检查。

chat template 把 messages 拼成一段带特殊标记的文字（Qwen2.5 的格式）：

  <|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n
  <|im_start|>user\n……规则、初始状态、目标……<|im_end|>\n
  <|im_start|>assistant\n
  unstack(A, D)\n……\nstack(C, D)<|im_end|>\n

  - 我们的样本没有 system 消息，模板自动补上第一行这句默认的话。训练和评测用同一个模板，两边一致。
  - 前三行是“提示”：模型要看到，但不用学着把它说出来。
  - 第四行是要学的回答，连同结束符 <|im_end|>：学会它，模型写完计划才知道停下。
  - 最后那个 \n 是模板在结束符后面补的。生成时一遇到 <|im_end|> 就停，永远轮不到它，所以不学。

labels 和 input_ids 一样长，一一对应。不参与 loss 的位置写 -100（PyTorch 的 cross_entropy 会跳过 -100）。
mask 不是删除：被 mask 的 token 仍然在 input_ids 里，模型照样看得到，只是不因为它们算 loss。

运行：.venv/bin/python sft/data.py
"""
import json
import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "blocksworld" / "v1"
# 这份模型属于另一个项目，只读加载，不往里面写任何文件。换位置时设置环境变量 QWEN_0_5B
MODEL_PATH = os.environ.get("QWEN_0_5B",
                            "/Users/luchen/Desktop/Idea/R_TPE_RL_mismatch/c1_pilot/models/qwen2.5-0.5b-instruct")
IGNORE = -100


def load_jsonl(name, limit=None):
    """读 data/blocksworld/v1/<name>.jsonl 的前 limit 条。文件生成时已经打乱过，取前几条不会偏向某一类题。"""
    rows = []
    with open(DATA_DIR / f"{name}.jsonl") as f:
        for line in f:
            if limit is not None and len(rows) == limit:
                break
            rows.append(json.loads(line))
    return rows


def encode(sample, tokenizer):
    """返回 (input_ids, labels)，两个一样长的整数列表。"""
    messages = sample["messages"]
    prompt_ids = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True)  # 到 <|im_start|>assistant\n 为止
    full_ids = tokenizer.apply_chat_template(messages)  # 连同回答的完整序列
    # 提示必须正好是完整序列的开头，下面才能按长度切开
    assert full_ids[:len(prompt_ids)] == prompt_ids
    # 完整序列最后两个 token 应该是 <|im_end|> 和 \n
    assert full_ids[-2] == tokenizer.convert_tokens_to_ids("<|im_end|>") and tokenizer.decode(full_ids[-1:]) == "\n"
    answer_ids = full_ids[len(prompt_ids):-1]  # 回答 + <|im_end|>，不含最后的 \n
    labels = [IGNORE] * len(prompt_ids) + answer_ids + [IGNORE]
    return full_ids, labels


def collate(batch, pad_id, width=None):
    """把一批长短不一的样本补成一样长（在右边补）。不给 width 就补到这批里最长的那条。
    补出来的位置：attention_mask = 0（模型不看它们），labels = -100（不算 loss）。"""
    width = width or max(len(ids) for ids, _ in batch)
    assert all(len(ids) <= width for ids, _ in batch), "有样本比 width 还长，会被截断"
    input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in batch]
    labels = [lab + [IGNORE] * (width - len(lab)) for _, lab in batch]
    attention_mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attention_mask)


def show_sample(sample, tokenizer):
    """打印一条样本：完整文字，以及逐个 token 是否参与 loss。题目中间部分太长，省略不打。"""
    ids, labels = encode(sample, tokenizer)
    print("====== chat template 拼出来的完整文字 ======")
    print(tokenizer.decode(ids))
    print("====== 逐个 token：学 = 参与 loss，· = 只看不学（label 是 -100） ======")
    print(" 位置   token id  学?  token 文字")
    first = next(i for i, lab in enumerate(labels) if lab != IGNORE)  # 第一个要学的 token
    for i in list(range(5)) + [None] + list(range(first - 6, len(ids))):
        if i is None:
            print("   ……（中间都是题目，全部是 ·）")
            continue
        mark = "学" if labels[i] != IGNORE else "·"
        print(f"{i:5d} {ids[i]:9d}   {mark}   {tokenizer.decode([ids[i]])!r}")
    n_learn = sum(lab != IGNORE for lab in labels)
    print(f"共 {len(ids)} 个 token：参与 loss 的 {n_learn} 个，只看不学的 {len(ids) - n_learn} 个")


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    train = load_jsonl("train")
    print("样本", train[0]["id"], f"（{train[0]['n_blocks']} 块积木，{train[0]['goal_type']}，最短 {train[0]['plan_len']} 步）\n")
    show_sample(train[0], tokenizer)

    print("\n====== 整个 train / val 的长度（每条都过一遍 encode 里的检查） ======")
    for name in ("train", "val"):
        encoded = [encode(s, tokenizer) for s in load_jsonl(name)]
        lengths = [len(ids) for ids, _ in encoded]
        learn = [sum(lab != IGNORE for lab in labels) for _, labels in encoded]
        print(f"{name}: {len(encoded)} 条，总长 {min(lengths)}–{max(lengths)} 个 token（平均 {sum(lengths) / len(lengths):.0f}），"
              f"参与 loss 的 {min(learn)}–{max(learn)} 个（平均 {sum(learn) / len(learn):.0f}）")


if __name__ == "__main__":
    main()
