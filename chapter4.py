"""
Chapter 4: 多层感知机与正则化。

内容按学习顺序整理:
1. MLP 从零实现
2. MLP 简洁实现
3. 多项式回归
4. 权重衰减
5. Dropout

"""

import math

import numpy as np
import torch
from torch import nn

import mini_d2l as d2l


# -----------------------------
# 1. MLP 从零实现
# -----------------------------
def relu(X):
    """ReLU 激活函数。"""
    return torch.max(X, torch.zeros_like(X))


def run_mlp_scratch():
    """运行多层感知机从零实现。"""
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs))
    params = [W1, b1, W2, b2]

    def net(X):
        X = X.reshape((-1, num_inputs))
        H = relu(torch.matmul(X, W1) + b1)
        return torch.matmul(H, W2) + b2

    loss = nn.CrossEntropyLoss()
    num_epochs = 10
    lr = 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


# -----------------------------
# 2. MLP 简洁实现
# -----------------------------
def init_linear_weights(m):
    """只对线性层做正态初始化。"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


def run_mlp_concise():
    """运行多层感知机简洁实现。"""
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    net.apply(init_linear_weights)

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# -----------------------------
# 3. 多项式回归
# -----------------------------
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上的平均损失。"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def build_polynomial_data(max_degree=20, n_train=100, n_test=100):
    """生成多项式回归实验数据。"""
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)

    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)

    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    true_w, features, poly_features, labels = [
        torch.tensor(x, dtype=torch.float32)
        for x in [true_w, features, poly_features, labels]
    ]
    return true_w, features, poly_features, labels, n_train, n_test


def train_polynomial_model(
    train_features, test_features, train_labels, test_labels, num_epochs=400
):
    """训练一个多项式回归模型并可视化损失。"""
    loss = nn.MSELoss(reduction="none")
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])

    train_iter = d2l.load_array(
        (train_features, train_labels.reshape(-1, 1)), batch_size
    )
    test_iter = d2l.load_array(
        (test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False
    )
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(
        xlabel="epoch",
        ylabel="loss",
        yscale="log",
        xlim=[1, num_epochs],
        ylim=[1e-3, 1e2],
        legend=["train", "test"],
    )

    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(
                epoch + 1,
                (
                    evaluate_loss(net, train_iter, loss),
                    evaluate_loss(net, test_iter, loss),
                ),
            )


def run_polynomial_regression():
    """依次演示欠拟合、正常拟合和过拟合。"""
    _, _, poly_features, labels, n_train, _ = build_polynomial_data()

    train_polynomial_model(
        poly_features[:n_train, :4],
        poly_features[n_train:, :4],
        labels[:n_train],
        labels[n_train:],
    )
    train_polynomial_model(
        poly_features[:n_train, :2],
        poly_features[n_train:, :2],
        labels[:n_train],
        labels[n_train:],
    )
    train_polynomial_model(
        poly_features[:n_train, :],
        poly_features[n_train:, :],
        labels[:n_train],
        labels[n_train:],
        num_epochs=1500,
    )


# -----------------------------
# 4. 权重衰减
# -----------------------------
def get_weight_decay_data(num_train=20, num_test=100, num_inputs=200):
    """生成高维线性回归数据。"""
    true_w = torch.ones((num_inputs, 1)) * 0.01
    true_b = 0.05
    train_data = d2l.synthetic_data(true_w, true_b, num_train)
    test_data = d2l.synthetic_data(true_w, true_b, num_test)
    train_iter = d2l.load_array(train_data, batch_size=5)
    test_iter = d2l.load_array(test_data, batch_size=5, is_train=False)
    return train_iter, test_iter, num_inputs


def init_params(num_inputs):
    """初始化线性模型参数。"""
    W = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [W, b]


def l2_penalty(W):
    """L2 正则项。"""
    return torch.sum(W.pow(2)) / 2


def train_weight_decay_scratch(lambd):
    """从零实现权重衰减。"""
    train_iter, test_iter, num_inputs = get_weight_decay_data()
    W, b = init_params(num_inputs)
    net, loss = lambda X: d2l.linreg(X, W, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(
        xlabel="epochs",
        ylabel="loss",
        yscale="log",
        xlim=[5, num_epochs],
        legend=["train", "test"],
    )

    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(W)
            l.sum().backward()
            d2l.sgd([W, b], lr, X.shape[0])
        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,
                (
                    d2l.evaluate_loss(net, train_iter, loss),
                    d2l.evaluate_loss(net, test_iter, loss),
                ),
            )


def train_weight_decay_concise(wd):
    """使用优化器参数实现权重衰减。"""
    train_iter, test_iter, num_inputs = get_weight_decay_data()
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()

    loss = nn.MSELoss(reduction="none")
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD(
        [
            {"params": net[0].weight, "weight_decay": wd},
            {"params": net[0].bias},
        ],
        lr=lr,
    )
    animator = d2l.Animator(
        xlabel="epochs",
        ylabel="loss",
        yscale="log",
        xlim=[5, num_epochs],
        legend=["train", "test"],
    )

    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y.reshape(-1, 1))
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(
                epoch + 1,
                (
                    d2l.evaluate_loss(net, train_iter, loss),
                    d2l.evaluate_loss(net, test_iter, loss),
                ),
            )


def run_weight_decay():
    """演示不加正则和加权重衰减两种训练方式。"""
    train_weight_decay_scratch(lambd=0)
    train_weight_decay_scratch(lambd=3)
    train_weight_decay_concise(0)
    train_weight_decay_concise(3)


# -----------------------------
# 5. Dropout
# -----------------------------
def dropout_layer(X, dropout):
    """手动实现 dropout。"""
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape, device=X.device) > dropout).float()
    return mask * X / (1.0 - dropout)


class DropoutNet(nn.Module):
    """从零思路实现的带 dropout 的 MLP。"""

    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.num_inputs = num_inputs
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        self.dropout1 = 0.2
        self.dropout2 = 0.5

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H1 = self.relu(self.lin1(X))
        if self.training:
            H1 = dropout_layer(H1, self.dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout2)
        return self.lin3(H2)


def run_dropout_scratch():
    """运行从零实现的 dropout 网络。"""
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = DropoutNet(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction="none")
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


def run_dropout_concise():
    """运行 PyTorch 简洁版 dropout 网络。"""
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(num_hiddens2, num_outputs),
    )
    net.apply(init_linear_weights)

    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction="none")
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    # 按需取消注释，只运行当前学习的小节。
    # run_mlp_scratch()
    # run_mlp_concise()
    # run_polynomial_regression()
    # run_weight_decay()
    # run_dropout_concise()
    print("chapter4.py 已切换为不依赖 d2l。请按需取消注释对应示例。")
