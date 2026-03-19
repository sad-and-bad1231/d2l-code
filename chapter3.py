"""
Chapter 3: 线性回归与 softmax 回归。

本文件按 d2l 第 3 章的学习顺序整理:
1. 线性回归从零实现
2. 线性回归简洁实现
3. softmax 回归从零实现
4. softmax 回归简洁实现

"""
import random

import torch
from torch import nn
from torch.utils import data

import mini_d2l as d2l


# -----------------------------
# 1. 线性回归从零实现
# -----------------------------
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的人造数据集。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """按小批量随机打乱顺序读取数据。"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型。"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """平方损失。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def run_linear_regression_scratch():
    """运行线性回归从零实现示例。"""
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    batch_size = 10
    num_epochs = 3

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = squared_loss(linreg(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = squared_loss(linreg(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

    print(f"w的估计误差: {true_w - w.reshape(true_w.shape)}")
    print(f"b的估计误差: {true_b - b}")


# -----------------------------
# 2. 线性回归简洁实现
# -----------------------------
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch DataLoader。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def run_linear_regression_concise():
    """运行线性回归简洁实现示例。"""
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    train_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {float(l):f}")

    w = net[0].weight.data
    b = net[0].bias.data
    print("w的估计误差:", true_w - w.reshape(true_w.shape))
    print("b的估计误差:", true_b - b)


# -----------------------------
# 3. softmax 回归从零实现
# -----------------------------
class Accumulator:
    """在多个变量上累加。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def softmax(X):
    """对每一行做 softmax。"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition


def cross_entropy(y_hat, y):
    """交叉熵损失。"""
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """计算预测正确的样本数。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的准确率。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练一个 epoch。"""
    if isinstance(net, torch.nn.Module):
        net.train()

    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练第 3 章分类模型。"""
    animator = d2l.Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))


def predict_ch3(net, test_iter, n=6):
    """可视化前 n 个样本的真实标签和预测标签。"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


def run_softmax_regression_scratch():
    """运行 softmax 回归从零实现示例。"""
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 784
    num_outputs = 10
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def net(X):
        return softmax(torch.matmul(X.reshape((-1, num_inputs)), w) + b)

    lr = 0.1

    def updater(batch_size_):
        return d2l.sgd([w, b], lr, batch_size_)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)


# -----------------------------
# 4. softmax 回归简洁实现
# -----------------------------
def run_softmax_regression_concise():
    """运行 softmax 回归简洁实现示例。"""
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)

    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    predict_ch3(net, test_iter)


if __name__ == "__main__":
    # 按需取消注释，只运行你当前学习的部分。
    # run_linear_regression_scratch()
    # run_linear_regression_concise()
    # run_softmax_regression_scratch()
    # run_softmax_regression_concise()
    print("chapter3.py 已切换为不依赖 d2l。请按需取消注释对应示例。")
