"""
排查 MPS 内存：试运行时进程占到 77 GB，远超需要。这里只跑几十步训练，每步打印内存，分清是哪一种情况：
  allocated  张量实际占用的内存（还在用的）
  driver     MPS 一共向系统要了多少（allocated + 用完没还、留着备用的缓存）
  峰值        进程到目前为止用过的最大常驻内存（活动监视器显示的“内存”算法不同，数会不一样）
  allocated 稳定、driver 一直涨：缓存越攒越多，不是我们的代码泄漏
  allocated 也一直涨：有张量没被释放，才是真泄漏

每次只改一个条件，其余不变（同样的模型、同样的前 64 条 train、同样的 LoRA 设定）：
  --batch-size  每步几条（试运行用的是 8）
  --pad-to      不写 = 每批补到这批里最长的那条（试运行的做法，每步长度都不一样）；320 = 每批都补到 320
  --empty-cache 每步之后清一次 MPS 缓存
  --generate    训练完再按评测的做法（每批 16 条）生成一次，看生成阶段的内存

运行（要在 Claude Code 外面的终端里，沙箱里用不了 MPS）：
  .venv/bin/python sft/mem_probe.py                              # 和试运行一样
  .venv/bin/python sft/mem_probe.py --pad-to 320                 # 只改：固定长度
  .venv/bin/python sft/mem_probe.py --pad-to 320 --empty-cache   # 再加：清缓存
  .venv/bin/python sft/mem_probe.py --batch-size 1               # 只改：batch size
不写任何文件，只打印。
"""
import argparse
import resource

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from data import MODEL_PATH, collate, encode, load_jsonl
from eval_plans import generate
from train_lora import DEVICE, LORA, LR, SEED, load_base_model

GB = 1024 ** 3


def memory():
    """返回一行字：allocated / driver（只有 MPS 有）和进程到目前为止的内存峰值，单位 GB。"""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # macOS 上单位是字节
    if DEVICE == "mps":
        return (f"allocated {torch.mps.current_allocated_memory() / GB:6.2f}  "
                f"driver {torch.mps.driver_allocated_memory() / GB:6.2f}  峰值 {peak / GB:6.2f}")
    return f"峰值 {peak / GB:6.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pad-to", type=int, default=None)
    parser.add_argument("--empty-cache", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    print(f"设备 {DEVICE}，batch {args.batch_size}，补齐到 {args.pad_to or '本批最长'}，"
          f"清缓存 {'是' if args.empty_cache else '否'}，{args.steps} 步")

    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    train = [encode(s, tokenizer) for s in load_jsonl("train", 64)]
    print(f"[开始]         {memory()}")
    model = get_peft_model(load_base_model(), LoraConfig(task_type="CAUSAL_LM", **LORA))
    print(f"[加载模型后]   {memory()}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    model.train()
    for step in range(args.steps):
        start = step * args.batch_size % len(train)
        chunk = (train + train)[start:start + args.batch_size]  # 在前 64 条里循环取
        input_ids, labels, attention_mask = (t.to(DEVICE) for t in collate(chunk, tokenizer.pad_token_id, args.pad_to))
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        width = input_ids.shape[1]
        del input_ids, labels, attention_mask, loss
        if args.empty_cache and DEVICE == "mps":
            torch.mps.empty_cache()
        print(f"step {step + 1:3d}  长度 {width:3d}  {memory()}")
    if args.generate:
        generate(model, tokenizer, load_jsonl("val", 16))
        print(f"[生成 16 条后] {memory()}")


if __name__ == "__main__":
    main()
