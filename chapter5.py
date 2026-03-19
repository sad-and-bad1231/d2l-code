"""Chapter 5: 深度学习计算。

本文件保留 D2L 第 5 章的常见教学内容，但实现不依赖 d2l:
1. 层和块
2. 参数管理
3. 延后初始化与自定义层
4. 文件读写
5. GPU 设备使用
"""

import torch
from torch import nn
from torch.nn import functional as F

import mini_d2l as d2l


def build_mlp():
    return nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


class CenteredLayer(nn.Module):
    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


def inspect_parameters():
    net = build_mlp()
    X = torch.rand(2, 20)
    _ = net(X)
    print("state_dict keys:", list(net.state_dict().keys()))
    print("first layer weight shape:", net[0].weight.shape)
    print("first layer bias[:5]:", net[0].bias[:5].data)


def demo_custom_layers():
    X = torch.rand(2, 20)
    centered = CenteredLayer()
    print("centered mean:", float(centered(torch.tensor([1.0, 2.0, 3.0])).mean()))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    print("sequential centered output mean:", float(net(torch.rand(4, 8)).mean()))

    dense = MyLinear(5, 3)
    print("custom linear output shape:", dense(torch.rand(2, 5)).shape)


def demo_composition():
    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    X = torch.rand(2, 20)
    print("composed network output:", chimera(X))


def save_and_load_parameters():
    X = torch.randn(size=(2, 20))
    net = build_mlp()
    _ = net(X)
    torch.save(net.state_dict(), "mlp.params")

    clone = build_mlp()
    clone.load_state_dict(torch.load("mlp.params"))
    clone.eval()
    print("parameter file saved and reloaded successfully")


def demo_gpu():
    device = d2l.try_gpu()
    X = torch.ones((2, 3), device=device)
    net = build_mlp().to(device)
    print("device:", device)
    print("tensor device:", X.device)
    print("network output device:", net(X).device)


if __name__ == "__main__":
    # inspect_parameters()
    # demo_custom_layers()
    # demo_composition()
    # save_and_load_parameters()
    # demo_gpu()
    print("chapter5.py 已创建且不依赖 d2l。请按需取消注释对应示例。")
