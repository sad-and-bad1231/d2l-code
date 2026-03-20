"""Chapter 6: 卷积神经网络。

1. 二维互相关
2. 自定义卷积层
3. 填充与步幅辅助函数
4. 池化
5. LeNet
6. GPU / CPU 训练函数

"""

import torch
from torch import nn
import mini_d2l as d2l


# -----------------------------
# 1. 二维互相关
# -----------------------------
def accuracy(y_hat, y):
    """计算预测正确的样本数。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    """用参数形式封装手写二维卷积层。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def demo_corr2d():
    """演示边缘检测卷积核。"""
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    return X, Y


def learn_kernel():
    """通过梯度下降学习简单卷积核。"""
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)

    conv2d = Conv2D(kernel_size=(1, 2))
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        conv2d.bias.data[:] -= lr * conv2d.bias.grad
        if (i + 1) % 2 == 0:
            print(f"epoch {i + 1}, loss {l.sum():.3f}")


# -----------------------------
# 2. 填充、步幅、池化
# -----------------------------
def comp_conv2d(conv2d, X):
    """把二维输入临时扩成四维，便于直接喂给 Conv2d。"""
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])


def pool2d(X, pool_size, mode="max"):
    """手写二维池化层。"""
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1), device=X.device)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            region = X[i : i + p_h, j : j + p_w]
            if mode == "max":
                Y[i, j] = region.max()
            elif mode == "avg":
                Y[i, j] = region.mean()
            else:
                raise ValueError(f"unknown pooling mode: {mode}")
    return Y


def demo_pool2d():
    """演示最大池化与平均池化。"""
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), "avg"))


# -----------------------------
# 3. LeNet
# -----------------------------
def build_lenet():
    """构建 LeNet。"""
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )


def show_layer_shapes(net):
    """打印每一层的输出形状。"""
    X = torch.rand(size=(1, 1, 28, 28))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, "output shape:\t", X.shape)


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在指定设备上评估准确率。"""
    if isinstance(net, torch.nn.Module):
        net.eval()
        if device is None:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用 GPU 训练卷积网络。"""

    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        legend=["train loss", "train acc", "test acc"],
    )
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % max(1, (num_batches // 5)) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}")


def train_ch6_cpu(net, train_iter, test_iter, num_epochs, lr):
    """CPU 版本训练函数，便于没有 GPU 时复现。"""

    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    device = torch.device("cpu")
    net.to(device)
    print("training on", device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        metric = [0.0, 0.0, 0.0]
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric[0] += l.item() * X.shape[0]
                metric[1] += accuracy(y_hat, y)
                metric[2] += X.shape[0]

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        print(
            f"epoch {epoch + 1}: train loss {train_loss:.4f}, "
            f"train acc {train_acc:.4f}, test acc {test_acc:.4f}"
        )


def run_lenet(device=None):
    """训练 LeNet。"""
    if device is None:
        device = d2l.try_gpu()
    net = build_lenet()
    show_layer_shapes(net)
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)


def main():
    """默认只打印可选入口，避免直接运行时进入训练。"""
    print("Chapter 6 可用入口：")
    print("- demo_corr2d()")
    print("- learn_kernel()")
    print("- demo_pool2d()")
    print("- run_lenet()")


if __name__ == "__main__":
    main()
