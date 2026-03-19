"""Chapter 7: 现代卷积神经网络。

本文件整理了 d2l 第 7 章常见模型:
- AlexNet
- VGG
- NiN
- GoogLeNet
- BatchNorm
- ResNet
- DenseNet

"""

import torch
from torch import nn
from torch.nn import functional as F

import mini_d2l as d2l


# -----------------------------
# 1. AlexNet
# -----------------------------
def build_alexnet():
    """构建 AlexNet。"""
    return nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        nn.Linear(6400, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 10),
    )


# -----------------------------
# 2. VGG
# -----------------------------
def vgg_block(num_convs, in_channels, out_channels):
    """构建一个 VGG 块。"""
    layers = []
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def build_vgg(conv_arch):
    """根据卷积块配置构建 VGG。"""
    conv_blks = []
    in_channels = 1
    for num_convs, out_channels in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10),
    )


def default_vgg_arch():
    """VGG-11 配置。"""
    return ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


def small_vgg_arch(ratio=4):
    """缩小版 VGG，便于在普通机器上训练。"""
    return [(pair[0], pair[1] // ratio) for pair in default_vgg_arch()]


# -----------------------------
# 3. NiN
# -----------------------------
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """NiN block: 普通卷积后接两个 1x1 卷积。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )


def build_nin():
    """构建 NiN。"""
    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


# -----------------------------
# 4. GoogLeNet
# -----------------------------
class Inception(nn.Module):
    """GoogLeNet 的 Inception 模块。"""

    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


def build_googlenet():
    """构建 GoogLeNet。"""
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    return nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


# -----------------------------
# 5. BatchNorm
# -----------------------------
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """从零实现批量规范化。"""
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    """兼容全连接层和卷积层的手写 BatchNorm。"""

    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X,
            self.gamma,
            self.beta,
            self.moving_mean,
            self.moving_var,
            eps=1e-5,
            momentum=0.9,
        )
        return Y


def build_lenet_with_batchnorm():
    """LeNet + 手写 BatchNorm。"""
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        BatchNorm(120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNorm(84, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )


def build_lenet_with_torch_batchnorm():
    """LeNet + PyTorch BatchNorm。"""
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        nn.BatchNorm1d(120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),
        nn.Linear(84, 10),
    )


# -----------------------------
# 6. ResNet
# -----------------------------
class Residual(nn.Module):
    """ResNet 残差块。"""

    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = (
            nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
            if use_1x1conv
            else None
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """构建一组残差块。"""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(
                    input_channels,
                    num_channels,
                    use_1x1conv=True,
                    strides=2,
                )
            )
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def build_resnet18():
    """构建适配 Fashion-MNIST 的 ResNet-18 风格网络。"""
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    return nn.Sequential(
        b1,
        b2,
        b3,
        b4,
        b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 10),
    )


# -----------------------------
# 7. DenseNet
# -----------------------------
def conv_block(input_channels, num_channels):
    """DenseNet 的卷积块。"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
    )


class DenseBlock(nn.Module):
    """稠密块: 每层都接收前面所有层的输出。"""

    def __init__(self, num_convs, input_channels, num_channels):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    """转换层: 压缩通道数并减小空间尺寸。"""
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )


def build_densenet():
    """构建 DenseNet。"""
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []
    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        num_channels += num_convs * growth_rate
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    return nn.Sequential(
        b1,
        *blks,
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(num_channels, 10),
    )


# -----------------------------
# 8. 通用辅助函数
# -----------------------------
def show_layer_shapes(net, input_shape):
    """打印网络每一层输出形状。"""
    X = torch.randn(size=input_shape)
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, "output shape:\t", X.shape)


def train_model(net, batch_size, resize, lr, num_epochs, device=None):
    """统一训练入口。"""
    if device is None:
        device = d2l.try_gpu()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=resize)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)


if __name__ == "__main__":
    # 按需选择一个模型训练，不要一次性全部运行。
    # net = build_alexnet()
    # train_model(net, batch_size=128, resize=224, lr=0.01, num_epochs=10)

    # net = build_vgg(small_vgg_arch())
    # train_model(net, batch_size=128, resize=224, lr=0.05, num_epochs=10)

    # net = build_nin()
    # train_model(net, batch_size=128, resize=224, lr=0.1, num_epochs=10)

    # net = build_googlenet()
    # train_model(net, batch_size=128, resize=96, lr=0.1, num_epochs=10)

    # net = build_lenet_with_batchnorm()
    # train_model(net, batch_size=256, resize=None, lr=0.1, num_epochs=10)

    # net = build_resnet18()
    # train_model(net, batch_size=256, resize=96, lr=0.05, num_epochs=10)

    # net = build_densenet()
    # train_model(net, batch_size=256, resize=96, lr=0.1, num_epochs=10)
    print("chapter7.py 已切换为不依赖 d2l。请按需取消注释对应示例。")
