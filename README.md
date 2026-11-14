# Agent RL Lab

一个边写边学的 LLM 训练实验仓库：从文本任务规划入手，逐步理解 SFT、GRPO、DAPO、多轮 agent、记忆和可验证的自我改进。

## 当前状态

- 仓库初始化日期：2026-10-28。
- `blocksworld/` 已有模拟器、BFS 参考求解器、出题与去重代码；正式数据集和 6–7 块积木泛化题已在本地生成。模型、数据和训练输出按仓库约定不提交 Git。
- 已在 Mac 上完成 Qwen2.5-0.5B-Instruct 的 LoRA SFT，以及 20 步简化 GRPO 试运行；[实验记录](docs/lessons/03-sft-to-grpo.md)包括设置、完整验证集对比和真实案例。**这不是最终测试集或泛化能力评测。**
- 已实现积木任务的 DAPO 教学版，并进行了小规模真实模型冒烟检查；[记录](docs/lessons/05-dapo-first-implementation.md)区分单元测试、流程验证与尚未开展的性能实验。
- 已完成 20 批积木 DAPO 教学实验及完整 val 同题对比，见[实验记录](docs/lessons/06-dapo-first-run.md)；这不是论文复现或独立 test 结论。
- 已完成确定性多轮交互、多轮 SFT 和小规模多轮 RL 学习；随后加入逐步奖励、reward-to-go、价值预测与每步优势。[增加到 50 批](docs/lessons/16-joint50-training-budget.md)后，两组在 30 道开发题上分别成功 11 和 10 道；[策略学习率与梯度裁剪小试](docs/lessons/18-actor-lr-clip-pilot.md)仍需多随机种子复核，单次结果不足以判断方法普遍优劣。
- 多轮阶段的代码与测试仍在本地整理，暂未全部纳入公开仓库；下方学习记录公开的是过程、设置、结果及其局限，不包含模型权重和原始轨迹。
- 早期 `sft/train_lora.py` 使用固定输出目录，重复运行可能覆盖结果；全量 SFT 和 GRPO 脚本使用独立的新运行目录。
- 第一阶段使用 Blocksworld（积木世界）的离散任务规划，不控制真实机器人的电机，也不涉及图像策略或连续动作。
- 学习目标是能够解释并亲手修改训练链路；先跑通、理解，再扩大模型和任务。

## 从这里开始

1. [一条积木任务：先看输入、动作和判分](docs/lessons/01-one-blocksworld-task.md)
2. [第一次 SFT 与 MPS 内存排查](docs/lessons/02-first-sft-and-mps-memory.md)
3. [完整 SFT 到第一次 GRPO：实验记录与真实案例](docs/lessons/03-sft-to-grpo.md)
4. [GRPO 中断续训：保存什么、怎样验证](docs/lessons/04-grpo-resume.md)
5. [DAPO：四个机制与第一次冒烟测试](docs/lessons/05-dapo-first-implementation.md)
6. [DAPO：20 批训练与完整验证集对照](docs/lessons/06-dapo-first-run.md)
7. [多轮 SFT：错误反馈与闭环纠正](docs/lessons/07-multiturn-sft-and-recovery.md)
8. [多轮 GRPO：真实模型冒烟](docs/lessons/08-multiturn-grpo-smoke.md)
9. [多题多轮 GRPO：三批小试跑](docs/lessons/09-multiturn-grpo-pilot.md)
10. [真实 Qwen 的逐步价值头：一次链路冒烟](docs/lessons/10-qwen-value-head-smoke.md)
11. [多轮逐步奖励与价值头：训练和续训](docs/lessons/11-multiturn-step-value-training.md)
12. [逐步回报：价值头的 100 批对照](docs/lessons/12-multiturn-value-ablation.md)
13. [固定策略：10 步价值头小试](docs/lessons/13-frozen-policy-value-horizon10.md)
14. [固定开局：从 4 次采样增加到 16 次](docs/lessons/14-value-sampling-uncertainty.md)
15. [10 步多轮训练：价值头有没有额外收益](docs/lessons/15-joint10-value-ablation.md)
16. [增加训练预算：多轮对照从 12 批到 50 批](docs/lessons/16-joint50-training-budget.md)
17. [梯度裁剪、学习率和价值头：先检查再做多种子实验](docs/lessons/17-gradient-clip-lr-diagnostics.md)
18. [策略学习率与梯度裁剪：12 批受控小试](docs/lessons/18-actor-lr-clip-pilot.md)
19. [实践路线和验收条件](docs/ROADMAP.md)
20. [原有 RL 13 讲基础路线](docs/RL_FOUNDATIONS.md)
21. [讲解与协作约定](AGENTS.md)

实践路线：

**任务与验证器 → SFT → GRPO → DAPO → 确定性多轮交互 → 多轮训练数据与 RL → 有限自我训练 → 可学习记忆。**

DAPO 是与 GRPO 密切相关的改进方案，不是必然更好的替代品。RSI 是 Recursive Self-Improvement（递归自我改进）的研究方向，不是与 GRPO 并列的单一优化算法。最初只做有独立验证器和固定评测的有限自我训练循环，不声称已经实现开放式递归自我改进。

## 仓库边界

- 不把整个原工作目录或个人材料加入这个仓库。
- 模型、数据、日志和 checkpoint 不提交 Git；后续仅保存生成脚本、配置、版本信息及小型人工检查样例。
- 不把作者论文结果填写成我们的实验结果；没有运行的实验明确标为未运行。
- 不自动租用 GPU、开启付费服务或运行长期训练。
- 每次课程只推进一个可理解、可检查的小环节，不预先生成整套训练框架。
- 自动化测试还需要单独安装 `pytest`；它不包含在 `sft/requirements.txt` 的训练依赖中。

## 上游资料

- [DAPO 论文](https://arxiv.org/abs/2503.14476)与[项目](https://dapo-sia.github.io/)
- [ReST-EM：验证筛选后的迭代自我训练](https://arxiv.org/abs/2312.06585)
- [STaR](https://arxiv.org/abs/2203.14465)
- [Absolute Zero](https://arxiv.org/abs/2505.03335)
- [ALFWorld](https://alfworld.github.io/)
- [verl-agent / GiGPO 官方代码](https://github.com/langfengQ/verl-agent)
- [THUDM / zai-org AgentInstruct 数据卡](https://huggingface.co/datasets/zai-org/AgentInstruct)

这些链接是阅读入口，不表示已经选定版本、验证兼容性或完成论文复现。
