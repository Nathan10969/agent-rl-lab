# RL 基础：保留原来的 13 讲顺序

这是从当前对话整理的原有约定，不是新的课程安排，也不是跨对话的内置记忆。实践课可引用基础课，不要求重头讲完才能动手。

1. MDP：S、A、P、R、γ 分别是什么。
2. trajectory、return、discount。
3. V(s)、Q(s,a)、advantage。
4. Bellman 方程。
5. Monte Carlo 与 Temporal Difference。
6. policy gradient 从哪里推导出来。
7. REINFORCE 数值例子。
8. Actor–Critic。
9. PPO 的 ratio、clip、advantage。
10. LLM RLHF：reward model、KL、PPO。
11. GRPO、DPO 与 PPO 的关系；实践中补充 DAPO 对照。
12. Agentic RL：工具调用、长程轨迹、过程奖励、信用分配。
13. 用一个真正的小环境写出完整 RL 训练循环。

## 截至 2026-10-28 的对话进度

已经讨论策略梯度、baseline、GAE、TRPO 和 PPO，也逐行学习过 NumPy 的 GAE 代码。讨论过不等于已掌握，遇到具体代码继续用小例子复习。

当前进入训练实践：先看懂任务、数据和判分，再到 tokenizer、模板、mask 和 SFT。不能把旧笔记里“第一讲进行中”的历史状态当作当前进度。
