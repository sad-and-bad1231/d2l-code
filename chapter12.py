"""D2L 第 12 章代码整理版。

本文件聚焦“计算性能与多 GPU 训练”主线，目标是：
1. 保留教材里关于同步/异步、JIT、并行与多 GPU 的核心思想；
2. 给出最小但能运行的 scratch 多 GPU 训练主线；
3. 在 Colab 单 GPU 或无 GPU 环境下安全降级，不因环境不足直接报错；
4. 默认入口只做轻量检查，不自动启动长时间训练。
"""

from __future__ import annotations

import copy
import time

import torch
from torch import nn

import chapter7 as ch7
import mini_d2l as d2l


# ==================== 1. 设备与基础性能检查 ====================
def synchronize_device(device: torch.device) -> None:
    """仅在 CUDA 设备上执行同步，避免异步执行影响计时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def synchronize_devices(devices) -> None:
    """同步一组设备。"""
    for device in devices:
        synchronize_device(device)


def inspect_hardware():
    """打印当前环境中的设备信息。"""
    devices = d2l.try_all_gpus()
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    print("default device:", d2l.try_gpu())
    print("all devices:", devices)
    return devices


def benchmark_matmul(matrix_size: int = 1024, num_iters: int = 20):
    """比较 CPU 与默认设备上的矩阵乘法耗时。"""
    cpu = torch.device("cpu")
    device = d2l.try_gpu()
    A_cpu = torch.randn((matrix_size, matrix_size), device=cpu)
    B_cpu = torch.randn((matrix_size, matrix_size), device=cpu)

    with d2l.Benchmark("cpu matmul"):
        for _ in range(num_iters):
            C_cpu = A_cpu @ B_cpu
        _ = C_cpu.sum().item()

    if device.type != "cuda":
        print("当前没有 GPU，跳过 GPU matmul 对比。")
        return {"cpu_device": cpu, "target_device": device}

    A_gpu = A_cpu.to(device)
    B_gpu = B_cpu.to(device)
    synchronize_device(device)
    with d2l.Benchmark(f"{device} matmul"):
        for _ in range(num_iters):
            C_gpu = A_gpu @ B_gpu
        synchronize_device(device)
        _ = C_gpu.sum().item()

    return {"cpu_device": cpu, "target_device": device}


def benchmark_async_computation(matrix_size: int = 2048, num_iters: int = 10):
    """演示 GPU 上“提交算子”和“真正完成计算”之间的时间差。"""
    device = d2l.try_gpu()
    if device.type != "cuda":
        print("当前没有 GPU，跳过异步计算演示。")
        return None

    A = torch.randn((matrix_size, matrix_size), device=device)
    B = torch.randn((matrix_size, matrix_size), device=device)

    synchronize_device(device)
    start = time.time()
    for _ in range(num_iters):
        C = A @ B
    launch_time = time.time() - start

    synchronize_device(device)
    total_time = time.time() - start
    print(f"launch time without explicit sync: {launch_time:.4f} sec")
    print(f"total time with final sync: {total_time:.4f} sec")
    _ = C.sum().item()
    return {"launch_time": launch_time, "total_time": total_time}


# ==================== 2. JIT 与图优化 ====================
class TinyMLP(nn.Module):
    """用于 JIT 演示的极小前馈网络。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, X):
        return self.net(X)


def benchmark_jit(batch_size: int = 512, num_iters: int = 200):
    """比较 eager 与 trace 后模型在重复前向时的耗时。"""
    device = d2l.try_gpu()
    net = TinyMLP().to(device).eval()
    example = torch.randn((batch_size, 512), device=device)
    traced_net = torch.jit.trace(net, example)

    with torch.no_grad():
        synchronize_device(device)
        start = time.time()
        for _ in range(num_iters):
            y_eager = net(example)
        synchronize_device(device)
        eager_time = time.time() - start

        synchronize_device(device)
        start = time.time()
        for _ in range(num_iters):
            y_traced = traced_net(example)
        synchronize_device(device)
        traced_time = time.time() - start

    print(f"eager time: {eager_time:.4f} sec")
    print(f"traced time: {traced_time:.4f} sec")
    print("output shapes:", y_eager.shape, y_traced.shape)
    return {"eager_time": eager_time, "traced_time": traced_time}


# ==================== 3. 自动并行与数据切分 ====================
def run_auto_parallelism_demo(matrix_size: int = 2048):
    """在两张 GPU 上演示并行张量计算。"""
    devices = d2l.try_all_gpus()
    cuda_devices = [device for device in devices if device.type == "cuda"]
    if len(cuda_devices) < 2:
        print("当前不足两张 GPU，跳过自动并行演示。")
        return None

    x_gpu1 = torch.randn((matrix_size, matrix_size), device=cuda_devices[0])
    x_gpu2 = torch.randn((matrix_size, matrix_size), device=cuda_devices[1])

    synchronize_devices(cuda_devices[:2])
    start = time.time()
    y_gpu1 = x_gpu1 @ x_gpu1
    y_gpu2 = x_gpu2 @ x_gpu2
    synchronize_devices(cuda_devices[:2])
    elapsed = time.time() - start

    print(f"parallel computation on {cuda_devices[:2]}: {elapsed:.4f} sec")
    return y_gpu1, y_gpu2


def split_batch(X, y, devices):
    """把一个 batch 沿样本维切到多个设备。"""
    if len(devices) == 0:
        devices = [torch.device("cpu")]
    num_shards = min(len(devices), X.shape[0])
    X_shards = torch.chunk(X, num_shards)
    y_shards = torch.chunk(y, num_shards)
    return (
        [x.to(device) for x, device in zip(X_shards, devices[:num_shards])],
        [label.to(device) for label, device in zip(y_shards, devices[:num_shards])],
    )


def allreduce(data):
    """把多设备上的同形状张量求和，并把结果写回每个设备。"""
    if len(data) <= 1:
        return data

    with torch.no_grad():
        total = data[0].clone()
        for tensor in data[1:]:
            total += tensor.to(data[0].device)
        for i, tensor in enumerate(data):
            tensor.copy_(total.to(tensor.device))
    return data


# ==================== 4. 多 GPU 训练主线 ====================
def get_resnet18_for_ch12(num_classes: int = 10):
    """复用 Chapter 7 的 ResNet-18 风格网络。"""
    net = ch7.build_resnet18()
    if num_classes != 10:
        net[-1] = nn.Linear(512, num_classes)
    return net


def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)


def _sync_params(source_net, target_nets):
    with torch.no_grad():
        source_params = list(source_net.parameters())
        for target_net in target_nets:
            for source_param, target_param in zip(source_params, target_net.parameters()):
                target_param.copy_(source_param.to(target_param.device))


def _zero_grads(nets):
    for net in nets:
        for param in net.parameters():
            if param.grad is not None:
                param.grad.zero_()


def train_batch_ch12(nets, X, y, loss, lr, devices):
    """scratch 多 GPU 训练一个 batch。"""
    X_shards, y_shards = split_batch(X, y, devices)
    active_nets = nets[: len(X_shards)]
    _zero_grads(active_nets)

    total_loss, total_correct, total_examples = 0.0, 0.0, 0
    for net, X_part, y_part in zip(active_nets, X_shards, y_shards):
        net.train()
        y_hat = net(X_part)
        l = loss(y_hat, y_part)
        l.sum().backward()
        total_loss += float(l.sum())
        total_correct += float((y_hat.argmax(dim=1) == y_part).sum())
        total_examples += y_part.numel()

    if len(active_nets) > 1:
        for params in zip(*[net.parameters() for net in active_nets]):
            grads = [param.grad.data for param in params]
            allreduce(grads)

    with torch.no_grad():
        for param in active_nets[0].parameters():
            param -= lr * param.grad / total_examples

    _sync_params(active_nets[0], active_nets[1:])
    return total_loss, total_correct, total_examples


def train_ch12(net, train_iter, test_iter, num_epochs, lr, devices=None):
    """scratch 多 GPU 训练入口。"""
    devices = devices if devices is not None else d2l.try_all_gpus()
    if len(devices) == 0:
        devices = [torch.device("cpu")]

    net.apply(_init_weights)
    nets = [copy.deepcopy(net).to(device) for device in devices]
    _sync_params(nets[0], nets[1:])

    loss = nn.CrossEntropyLoss(reduction="none")
    animator = d2l.Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        legend=["train loss", "train acc", "test acc"],
    )
    timer = d2l.Timer()

    print("training scratch on", devices)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for X, y in train_iter:
            timer.start()
            l, acc, num_examples = train_batch_ch12(nets, X, y, loss, lr, devices)
            metric.add(l, acc, num_examples)
            timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(nets[0], test_iter, devices[0])
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        print(
            f"epoch {epoch + 1}: train loss {train_l:.4f}, "
            f"train acc {train_acc:.4f}, test acc {test_acc:.4f}"
        )

    speed = metric[2] * num_epochs / timer.sum()
    print(f"{speed:.1f} examples/sec on {devices}")
    return {"animator": animator, "net": nets[0], "devices": devices}


def train_concise_ch12(net, train_iter, test_iter, num_epochs, lr, devices=None):
    """使用 `nn.DataParallel` 的简洁多 GPU 训练入口。"""
    devices = devices if devices is not None else d2l.try_all_gpus()
    if len(devices) == 0:
        devices = [torch.device("cpu")]
    main_device = devices[0]

    net.apply(_init_weights)
    if main_device.type == "cuda" and len(devices) > 1:
        device_ids = [device.index for device in devices if device.type == "cuda"]
        net = nn.DataParallel(net, device_ids=device_ids).to(main_device)
        print("training concise with DataParallel on", devices)
    else:
        net = net.to(main_device)
        print("当前不足多 GPU，concise 训练退化为单设备模式:", main_device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        legend=["train loss", "train acc", "test acc"],
    )
    timer = d2l.Timer()

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for X, y in train_iter:
            timer.start()
            optimizer.zero_grad()
            X = X.to(main_device)
            y = y.to(main_device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(float(l) * X.shape[0], float((y_hat.argmax(dim=1) == y).sum()), X.shape[0])
            timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, main_device)
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        print(
            f"epoch {epoch + 1}: train loss {train_l:.4f}, "
            f"train acc {train_acc:.4f}, test acc {test_acc:.4f}"
        )

    speed = metric[2] * num_epochs / timer.sum()
    print(f"{speed:.1f} examples/sec on {devices}")
    return {"animator": animator, "net": net, "devices": devices}


# ==================== 5. 运行入口 ====================
def inspect_split_batch():
    """用一个小 batch 验证数据切分逻辑。"""
    devices = d2l.try_all_gpus()
    X = torch.randn(8, 1, 28, 28)
    y = torch.arange(8)
    X_shards, y_shards = split_batch(X, y, devices)
    print("split batch shard shapes:", [tuple(x.shape) for x in X_shards])
    print("split label shard shapes:", [tuple(label.shape) for label in y_shards])
    return X_shards, y_shards


def main():
    """默认只做轻量检查。"""
    inspect_hardware()
    inspect_split_batch()
    benchmark_matmul(matrix_size=256, num_iters=5)
    benchmark_async_computation(matrix_size=512, num_iters=2)
    benchmark_jit(batch_size=128, num_iters=20)
    run_auto_parallelism_demo(matrix_size=256)

    # 以下训练较慢，按需取消注释。
    # train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=256, resize=96)
    # net = get_resnet18_for_ch12()
    # train_ch12(net, train_iter, test_iter, num_epochs=2, lr=0.2)
    # net = get_resnet18_for_ch12()
    # train_concise_ch12(net, train_iter, test_iter, num_epochs=2, lr=0.2)


if __name__ == "__main__":
    main()
