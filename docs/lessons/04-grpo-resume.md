# GRPO 续训：把训练现场也存下来

## 为什么只存 LoRA 不够？

LoRA adapter 记录模型参数，可用来加载并评测。但 AdamW 还维护历史梯度的统计；采样还依赖随机数状态；训练还需要知道已经用过哪些题。只加载 LoRA 并重新创建优化器，相当于从这些权重开始一个新训练过程，不是精确恢复中断现场。

历史 `outputs/grpo/trial-01/checkpoint-0010` 和 `checkpoint-0020` 只有 adapter，无法事后补回当时的优化器和随机状态。它们没有被覆盖，仍可以照旧用来评测。

## 新存档保存了什么？

[`resume_state.py`](../../grpo/resume_state.py) 在训练 step 完成、梯度清理后保存：

- adapter 权重和配置；
- AdamW 状态，包括参数组和梯度历史；
- 完成的 step、题目顺序、LoRA 参数名和顺序；
- Python、PyTorch CPU 及 MPS 随机数状态；
- 格式版本、数据／模型／tokenizer／代码指纹及影响训练的配置。

先写新建的临时目录，计算文件 SHA-256 并写完整标记，再发布为 checkpoint；已经存在的目标不会被主动覆盖。恢复时会核对标记、文件指纹和配置；只含 adapter 的旧存档会被拒绝。指纹用于发现文件变化，不证明文件来源可信；只从可信来源加载训练状态。

[`train_grpo.py`](../../grpo/train_grpo.py) 中恢复顺序是：加载基础模型和已保存 adapter → 重建 AdamW → 恢复优化器和题目顺序 → 最后恢复随机数状态 → 从下一步开始采样。可训练参数名称与顺序不一致会被拒绝。断点只支持完整 step 边界，不支持恢复半个 batch 的中间梯度。

## 本机对照：连续跑与续训

使用同一台 Mac、Qwen2.5-0.5B-Instruct、SFT step 378 adapter、相同训练数据、seed 0、MPS float32、每步 4 题、每题 8 答案：

1. 连续线：运行第 1–4 步。
2. 中断线：运行第 1–2 步并保存完整存档；启动新进程，从存档恢复第 3–4 步。
3. 不同运行写入不同的新目录，不使用验证集或 test。

逐项比较第 3–4 步，两个进程各有 64 个答案：题目 ID、生成 token、文本、奖励、优势全部一致。第 2 步存档的 LoRA 与完整状态也逐位一致；第 4 步的 LoRA 张量逐位一致、最大绝对差 0；AdamW 状态一致、张量最大绝对差 0；保存的 Python、CPU 和 MPS 随机状态也一致。

第一次对照中，token、参数和优化器已一致，但 Python 全局随机状态不同。进一步检查发现两个独立进程在第 2 步时该状态就不同：旧训练代码只固定了 PyTorch seed 和用于打乱顺序的独立 Python 随机对象，没有固定 Python 全局状态。当前训练路径没有用该全局状态生成答案，但为了完整记录，增加了全局初始化；新一轮对照才得到上述全部一致的结果。第一次结果也保留在本地，没有覆盖。

复核还发现需要防止误判的问题：两边同时漏记录时不能只比较相等；不能拿同一运行与自身比较或忽略续训来源；独立 `chat_template.jinja` 可以改变输入，必须纳入 tokenizer 指纹；LoRA 配置变化也可能改变行为，不能只比较权重。我们补了检查和测试，并增加发布前 adapter 文件可读取性检查；下列 `report-final-v2.json` 由改进后的代码重新运行生成。

该结果只说明**本次机器、软件、配置和短运行**的实际观察；不保证其他设备、版本或更长训练逐位相同，也不说明模型能力提高。CPU 自动化测试覆盖状态恢复、参数顺序不匹配、旧存档、篡改、拒绝覆盖、模板指纹变化、对照完整性和来源。

## 使用方式

确保基础模型和本地训练集可用。每次 `--run-dir` 必须是不存在的新目录；不要照抄已存在的路径。`--steps` 始终是总目标步数，`--stop-after` 是本次人为停止的位置：

```bash
.venv/bin/python -u grpo/train_grpo.py --run-dir outputs/grpo/MY-NEW-PART1 --steps 4 --save-every 2 --stop-after 2
.venv/bin/python -u grpo/train_grpo.py --run-dir outputs/grpo/MY-NEW-PART2 --steps 4 --save-every 2 --resume-from outputs/grpo/MY-NEW-PART1/checkpoint-0002
```

续训的 `--steps`、保存频率和影响训练的设置需与存档兼容；若代码或环境指纹改变，程序会拒绝作为同一实验恢复。续训的本段日志和计数仅覆盖新进程执行的步骤，不会伪称包含中断前的全部日志；本段 `lora_drift_from_start` 以该进程加载时的 adapter 为参考点。

## 本地结果与备份

实验输出和权重在 `outputs/`，被 Git 忽略，提交文档不等于备份。对照报告和最终 adapter 的 SHA-256：

| 本机文件 | SHA-256 |
| --- | --- |
| `outputs/grpo/resume-check-20261102-report.json`（第一次） | `f0a1e348d9a62669c18b1ca5d7efe89be4f94ef0069f0758f687b0d85fd38169` |
| `outputs/grpo/resume-check-20261102-report-v2.json`（第二次） | `d1f4cdb8c14abb779680554309193aecab67af4bb4b5aabd11a8131b57b109fb` |
| `outputs/grpo/resume-check-20261102-d/checkpoint-0004/adapter_model.safetensors` | `049f7e32c0ba78b2ff39040a95cfa775128fdf93df4e406b445af1cbdccc06a1` |
| `outputs/grpo/resume-check-20261102-f/checkpoint-0004/adapter_model.safetensors` | `049f7e32c0ba78b2ff39040a95cfa775128fdf93df4e406b445af1cbdccc06a1` |
| `outputs/grpo/resume-check-20261102-report-final.json`（复核后第一轮） | `e0155ec57a2e07aac296ae23e85ad21a90d26ed6352119988c289a5cca6633e9` |
| `outputs/grpo/resume-check-20261102-g/checkpoint-0004/adapter_model.safetensors`（连续） | `049f7e32c0ba78b2ff39040a95cfa775128fdf93df4e406b445af1cbdccc06a1` |
| `outputs/grpo/resume-check-20261102-i/checkpoint-0004/adapter_model.safetensors`（续训） | `049f7e32c0ba78b2ff39040a95cfa775128fdf93df4e406b445af1cbdccc06a1` |
| `outputs/grpo/resume-check-20261102-report-final-v2.json`（最终代码） | `e4021b299b503c2a1c111d01372729f783282e72930837ab96abd8f446096ffb` |
| `outputs/grpo/resume-check-20261102-j/checkpoint-0004/adapter_model.safetensors`（连续） | `049f7e32c0ba78b2ff39040a95cfa775128fdf93df4e406b445af1cbdccc06a1` |
| `outputs/grpo/resume-check-20261102-l/checkpoint-0004/adapter_model.safetensors`（续训） | `049f7e32c0ba78b2ff39040a95cfa775128fdf93df4e406b445af1cbdccc06a1` |
