"""D2L 第 11 章代码整理版。

本文件聚焦“优化算法”这一主线，按前面章节相同的整理风格保留：
1. 各优化器的从零实现与 PyTorch 简洁实现；
2. 统一的训练骨架，便于横向比较不同优化器；
3. 少量学习率调度器示例，但不展开整章所有实验；
4. 默认入口只做轻量检查，避免在 CPU 或离线环境下误触发长时间训练。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

import mini_d2l as d2l


# ==================== 1. 数据与基础训练骨架 ====================
d2l.DATA_HUB["airfoil"] = (
    d2l.DATA_URL + "airfoil_self_noise.dat",
    "76e5be1548fd8222e5074cf0faae75edff8cf93f",
)


def get_data_ch11(batch_size: int = 10, n: int = 1500):
    """下载空气动力学数据集并返回训练迭代器。"""
    data_file = d2l.download("airfoil")
    rows = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append([float(x) for x in stripped.split("\t")])

    data = torch.tensor(rows, dtype=torch.float32)

    # 标准化每一列，避免不同量纲导致训练不稳定。
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, unbiased=False, keepdim=True)
    std[std == 0] = 1
    data = (data - mean) / std
    data = data[:n]

    features = data[:, :-1]
    labels = data[:, -1].reshape(-1, 1)
    return d2l.load_array((features, labels), batch_size), features.shape[1]


def linreg(X, w, b):
    """线性回归模型。"""
    return X @ w + b


def squared_loss(y_hat, y):
    """平方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def init_ch11_params(feature_dim: int):
    """初始化第 11 章线性回归参数。"""
    w = torch.normal(0, 0.01, size=(feature_dim, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def evaluate_loss_ch11(net, data_iter, loss):
    """计算当前模型在整个数据集上的平均损失。"""
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            l = loss(net(X), y)
            metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train_ch11(
    trainer_fn: Callable,
    states,
    hyperparams: dict,
    data_iter,
    feature_dim: int,
    num_epochs: int = 2,
):
    """第 11 章 scratch 优化器统一训练入口。

    参数说明：
    - trainer_fn: 例如 `sgd`、`sgd_momentum`、`adam`
    - states: 优化器状态；若无需状态可传 `None`
    - hyperparams: 优化器超参数字典，至少应包含 `lr`
    - data_iter: 训练数据迭代器
    - feature_dim: 输入特征维度
    """
    w, b = init_ch11_params(feature_dim)
    net = lambda X: linreg(X, w, b)
    loss = squared_loss
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], legend=["train"])
    timer = d2l.Timer()
    total_examples = 0

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            batch_size = X.shape[0]
            metric.add(l.item() * batch_size, batch_size)
            total_examples += batch_size
        epoch_loss = evaluate_loss_ch11(net, data_iter, loss)
        animator.add(epoch + 1, (epoch_loss,))
        print(f"epoch {epoch + 1}, loss {epoch_loss:.4f}")

    speed = total_examples / timer.stop()
    print(f"train loss {epoch_loss:.4f}, {speed:.1f} examples/sec")
    return {"animator": animator, "loss": epoch_loss, "speed": speed, "params": (w, b)}


def train_concise_ch11(
    trainer_cls,
    hyperparams: dict,
    data_iter,
    feature_dim: int,
    num_epochs: int = 2,
):
    """第 11 章 PyTorch optimizer 统一训练入口。"""
    net = nn.Sequential(nn.Linear(feature_dim, 1))
    nn.init.normal_(net[0].weight, mean=0, std=0.01)
    nn.init.zeros_(net[0].bias)
    loss = nn.MSELoss(reduction="mean")

    if isinstance(trainer_cls, torch.optim.Optimizer):
        trainer = trainer_cls
    else:
        trainer = trainer_cls(net.parameters(), **hyperparams)

    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], legend=["train"])
    timer = d2l.Timer()
    total_examples = 0

    for epoch in range(num_epochs):
        for X, y in data_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
            total_examples += X.shape[0]
        epoch_loss = evaluate_loss_ch11(net, data_iter, lambda y_hat, y: (y_hat - y) ** 2 / 2)
        animator.add(epoch + 1, (epoch_loss,))
        print(f"epoch {epoch + 1}, loss {epoch_loss:.4f}")

    speed = total_examples / timer.stop()
    print(f"train loss {epoch_loss:.4f}, {speed:.1f} examples/sec")
    return {"animator": animator, "loss": epoch_loss, "speed": speed, "net": net}


def sgd(params, states, hyperparams):
    """从零实现小批量随机梯度下降。"""
    del states
    lr = hyperparams["lr"]
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()


# ==================== 2. 动量法 ====================
def init_momentum_states(feature_dim: int):
    """为 w 和 b 初始化速度变量 v。"""
    return (
        torch.zeros((feature_dim, 1)),
        torch.zeros(1),
    )


def sgd_momentum(params, states, hyperparams):
    """动量法：累计一个指数加权的历史梯度方向。"""
    lr = hyperparams["lr"]
    momentum = hyperparams["momentum"]
    with torch.no_grad():
        for param, velocity in zip(params, states):
            velocity[:] = momentum * velocity + param.grad
            param -= lr * velocity
            param.grad.zero_()


def run_momentum_scratch():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    states = init_momentum_states(feature_dim)
    return train_ch11(
        sgd_momentum,
        states,
        {"lr": 0.02, "momentum": 0.9},
        data_iter,
        feature_dim,
    )


def run_momentum_concise():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    return train_concise_ch11(
        torch.optim.SGD,
        {"lr": 0.02, "momentum": 0.9},
        data_iter,
        feature_dim,
    )


# ==================== 3. AdaGrad ====================
def init_adagrad_states(feature_dim: int):
    """累计梯度平方和 s。"""
    return (
        torch.zeros((feature_dim, 1)),
        torch.zeros(1),
    )


def adagrad(params, states, hyperparams):
    """AdaGrad：参数步长会随历史梯度平方和增大而减小。"""
    lr = hyperparams["lr"]
    eps = hyperparams.get("eps", 1e-6)
    with torch.no_grad():
        for param, state in zip(params, states):
            state[:] += param.grad ** 2
            param -= lr * param.grad / torch.sqrt(state + eps)
            param.grad.zero_()


def run_adagrad_scratch():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    states = init_adagrad_states(feature_dim)
    return train_ch11(adagrad, states, {"lr": 0.1}, data_iter, feature_dim)


def run_adagrad_concise():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    return train_concise_ch11(torch.optim.Adagrad, {"lr": 0.1}, data_iter, feature_dim)


# ==================== 4. RMSProp ====================
def init_rmsprop_states(feature_dim: int):
    """初始化梯度平方的指数加权平均。"""
    return (
        torch.zeros((feature_dim, 1)),
        torch.zeros(1),
    )


def rmsprop(params, states, hyperparams):
    """RMSProp：用指数加权平均代替 AdaGrad 的累积平方和。"""
    lr = hyperparams["lr"]
    gamma = hyperparams.get("gamma", 0.9)
    eps = hyperparams.get("eps", 1e-6)
    with torch.no_grad():
        for param, state in zip(params, states):
            state[:] = gamma * state + (1 - gamma) * (param.grad ** 2)
            param -= lr * param.grad / torch.sqrt(state + eps)
            param.grad.zero_()


def run_rmsprop_scratch():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    states = init_rmsprop_states(feature_dim)
    return train_ch11(
        rmsprop,
        states,
        {"lr": 0.01, "gamma": 0.9},
        data_iter,
        feature_dim,
    )


def run_rmsprop_concise():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    return train_concise_ch11(
        torch.optim.RMSprop,
        {"lr": 0.01, "alpha": 0.9},
        data_iter,
        feature_dim,
    )


# ==================== 5. Adadelta ====================
def init_adadelta_states(feature_dim: int):
    """Adadelta 同时维护梯度平方和参数更新平方的指数平均。"""
    return (
        torch.zeros((feature_dim, 1)),
        torch.zeros((feature_dim, 1)),
        torch.zeros(1),
        torch.zeros(1),
    )


def adadelta(params, states, hyperparams):
    """Adadelta：不直接使用全局学习率，而是用历史更新幅度自适应缩放。"""
    rho = hyperparams.get("rho", 0.9)
    eps = hyperparams.get("eps", 1e-5)
    with torch.no_grad():
        for i, param in enumerate(params):
            s = states[2 * i]
            delta = states[2 * i + 1]
            s[:] = rho * s + (1 - rho) * (param.grad ** 2)
            g = torch.sqrt(delta + eps) / torch.sqrt(s + eps) * param.grad
            param -= g
            delta[:] = rho * delta + (1 - rho) * (g ** 2)
            param.grad.zero_()


def run_adadelta_scratch():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    states = init_adadelta_states(feature_dim)
    return train_ch11(adadelta, states, {"rho": 0.9}, data_iter, feature_dim)


def run_adadelta_concise():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    return train_concise_ch11(
        torch.optim.Adadelta,
        {"rho": 0.9},
        data_iter,
        feature_dim,
    )


# ==================== 6. Adam ====================
def init_adam_states(feature_dim: int):
    """Adam 为每个参数维护一阶矩和二阶矩。"""
    return [
        [torch.zeros((feature_dim, 1)), torch.zeros((feature_dim, 1))],
        [torch.zeros(1), torch.zeros(1)],
    ]


def adam(params, states, hyperparams):
    """Adam：把动量法和 RMSProp 的思想合在一起。"""
    lr = hyperparams["lr"]
    beta1 = hyperparams.get("beta1", 0.9)
    beta2 = hyperparams.get("beta2", 0.999)
    eps = hyperparams.get("eps", 1e-6)
    hyperparams["t"] = hyperparams.get("t", 0) + 1
    t = hyperparams["t"]

    with torch.no_grad():
        for param, (v, s) in zip(params, states):
            v[:] = beta1 * v + (1 - beta1) * param.grad
            s[:] = beta2 * s + (1 - beta2) * (param.grad ** 2)
            v_hat = v / (1 - beta1 ** t)
            s_hat = s / (1 - beta2 ** t)
            param -= lr * v_hat / (torch.sqrt(s_hat) + eps)
            param.grad.zero_()


def run_adam_scratch():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    states = init_adam_states(feature_dim)
    return train_ch11(adam, states, {"lr": 0.01, "t": 0}, data_iter, feature_dim)


def run_adam_concise():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    return train_concise_ch11(torch.optim.Adam, {"lr": 0.01}, data_iter, feature_dim)


# ==================== 7. Yogi ====================
def init_yogi_states(feature_dim: int):
    """Yogi 与 Adam 类似，也维护一阶矩和二阶矩。"""
    return [
        [torch.zeros((feature_dim, 1)), torch.zeros((feature_dim, 1))],
        [torch.zeros(1), torch.zeros(1)],
    ]


def yogi(params, states, hyperparams):
    """Yogi：用符号控制二阶矩更新，减轻 Adam 在某些场景中过快增长的问题。"""
    lr = hyperparams["lr"]
    beta1 = hyperparams.get("beta1", 0.9)
    beta2 = hyperparams.get("beta2", 0.999)
    eps = hyperparams.get("eps", 1e-3)
    hyperparams["t"] = hyperparams.get("t", 0) + 1

    with torch.no_grad():
        for param, (v, s) in zip(params, states):
            grad = param.grad
            v[:] = beta1 * v + (1 - beta1) * grad
            s[:] += (1 - beta2) * torch.sign(grad ** 2 - s) * (grad ** 2)
            param -= lr * v / (torch.sqrt(s + eps))
            param.grad.zero_()


def run_yogi_scratch():
    data_iter, feature_dim = get_data_ch11(batch_size=10)
    states = init_yogi_states(feature_dim)
    return train_ch11(yogi, states, {"lr": 0.01, "t": 0}, data_iter, feature_dim)


# ==================== 8. 学习率调度器 ====================
class FactorScheduler:
    """每次调用都按固定因子衰减学习率。"""

    def __init__(self, factor: float = 1.0, stop_factor_lr: float = 1e-7, base_lr: float = 0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update: int) -> float:
        del num_update
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr


class CosineScheduler:
    """余弦调度器，可选预热阶段。"""

    def __init__(
        self,
        max_update: int,
        base_lr: float = 0.01,
        final_lr: float = 0.0,
        warmup_steps: int = 0,
        warmup_begin_lr: float = 0.0,
    ):
        self.max_update = max_update
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr

    def get_warmup_lr(self, num_update: int) -> float:
        return self.warmup_begin_lr + (
            self.base_lr - self.warmup_begin_lr
        ) * num_update / max(1, self.warmup_steps)

    def __call__(self, num_update: int) -> float:
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        if num_update > self.max_update:
            return self.final_lr

        progress = (num_update - self.warmup_steps) / max(1, self.max_update - self.warmup_steps)
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return self.final_lr + (self.base_lr - self.final_lr) * cosine


def demo_schedulers(num_steps: int = 10):
    """打印两个基础调度器在前若干步的学习率变化。"""
    factor_scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-4, base_lr=0.1)
    cosine_scheduler = CosineScheduler(
        max_update=max(1, num_steps - 1),
        base_lr=0.1,
        final_lr=0.01,
        warmup_steps=2,
        warmup_begin_lr=0.0,
    )

    factor_lrs = [factor_scheduler(step) for step in range(num_steps)]
    cosine_lrs = [cosine_scheduler(step) for step in range(num_steps)]
    print("FactorScheduler:", [round(lr, 5) for lr in factor_lrs])
    print("CosineScheduler:", [round(lr, 5) for lr in cosine_lrs])
    return factor_lrs, cosine_lrs


# ==================== 9. 运行入口 ====================
def inspect_ch11_data():
    """只取一批数据，检查维度是否合理。"""
    data_iter, feature_dim = get_data_ch11(batch_size=8, n=32)
    X, y = next(iter(data_iter))
    print(f"feature_dim={feature_dim}, X.shape={tuple(X.shape)}, y.shape={tuple(y.shape)}")


def main():
    """默认只做轻量检查。"""
    demo_schedulers(num_steps=8)
    try:
        inspect_ch11_data()
    except Exception as err:
        print(f"跳过空气动力学数据检查：{err}")
        print("如需运行该部分，请确认当前环境可联网，并允许在项目目录下创建 data 缓存。")

    # 以下训练在 CPU 上会稍慢，按需取消注释。
    # data_iter, feature_dim = get_data_ch11(batch_size=10)
    # train_ch11(sgd, None, {"lr": 0.03}, data_iter, feature_dim)
    # train_concise_ch11(torch.optim.SGD, {"lr": 0.03}, data_iter, feature_dim)
    # run_momentum_scratch()
    # run_momentum_concise()
    # run_adagrad_scratch()
    # run_adagrad_concise()
    # run_rmsprop_scratch()
    # run_rmsprop_concise()
    # run_adadelta_scratch()
    # run_adadelta_concise()
    # run_adam_scratch()
    # run_adam_concise()
    # run_yogi_scratch()


if __name__ == "__main__":
    main()
