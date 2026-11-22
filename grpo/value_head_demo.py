"""单个状态的价值头手算演示：只用 CPU，不加载模型或数据集。"""
import torch


def run_demo():
    # 同一个初始状态，两次采样得到的回报；z 是假设的隐藏表示。
    z = torch.tensor([2.0, 1.0], dtype=torch.float64)
    returns = torch.tensor([0.71, 0.90], dtype=torch.float64)

    # 这三个数是价值头要学习的参数，不是策略模型的 LoRA 参数。
    w1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    w2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([w1, w2, b], lr=0.05)

    optimizer.zero_grad()
    prediction = w1 * z[0] + w2 * z[1] + b
    loss = ((prediction - returns) ** 2).mean()
    prediction_before, loss_before = prediction.item(), loss.item()

    loss.backward()
    gradients = {"w1": w1.grad.item(), "w2": w2.grad.item(), "b": b.grad.item()}
    optimizer.step()

    with torch.no_grad():
        new_prediction = w1 * z[0] + w2 * z[1] + b
        new_loss = ((new_prediction - returns) ** 2).mean()

    return {
        "prediction_before": prediction_before,
        "loss_before": loss_before,
        "gradients": gradients,
        "parameters_after": {"w1": w1.item(), "w2": w2.item(), "b": b.item()},
        "prediction_after": new_prediction.item(),
        "loss_after": new_loss.item(),
    }


if __name__ == "__main__":
    result = run_demo()
    for name, value in result.items():
        print(f"{name}: {value}")
