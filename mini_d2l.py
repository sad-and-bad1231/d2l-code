"""面向 D2L 学习代码的轻量工具层。

目标：
- 保留教材中的核心训练流程与数据处理思路；
- 避免依赖 d2l 第三方包，便于在本地、Colab、Kaggle 复用；
- 只实现当前章节真正需要的最小功能集合。
"""

from __future__ import annotations

import collections
import hashlib
import math
import os
import random
import re
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_HUB: dict[str, tuple[str, str]] = {}
DATA_DIR = Path(__file__).resolve().parent / "data"


def try_gpu(i: int = 0) -> torch.device:
    """优先返回 GPU，否则退回 CPU。"""
    if torch.cuda.device_count() > i:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


class Timer:
    """记录累计耗时。"""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.start()

    def start(self) -> None:
        self.tik = time.time()

    def stop(self) -> float:
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self) -> float:
        return sum(self.times)


class Accumulator:
    """在多个变量上做累加。"""

    def __init__(self, n: int) -> None:
        self.data = [0.0] * n

    def add(self, *args) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self) -> None:
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]


class Animator:
    """简化版训练曲线记录器。

    脚本环境不强依赖实时显示，仅在内存中记录数据；
    如需查看曲线，可显式调用 `plot()`。
    """

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
    ):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.X: list[float] = []
        self.Y: list[list[float]] = []

    def add(self, x, y) -> None:
        if not isinstance(y, (list, tuple)):
            y = [y]
        if not self.Y:
            self.Y = [[] for _ in y]
        if isinstance(x, (list, tuple)):
            xs = list(x)
        else:
            xs = [x] * len(y)
        for i, (x_value, value) in enumerate(zip(xs, y)):
            if value is None:
                continue
            if i >= len(self.Y):
                self.Y.append([])
            while len(self.X) <= len(self.Y[i]) - 1:
                self.X.append(None)
            self.Y[i].append(float(value))
        if not self.X or self.X[-1] != x:
            self.X.append(x)

    def plot(self) -> None:
        if not self.Y:
            return
        for idx, series in enumerate(self.Y):
            xs = [x for x in self.X[: len(series)] if x is not None]
            label = self.legend[idx] if self.legend and idx < len(self.legend) else None
            plt.plot(xs, series, label=label)
        if self.xlabel:
            plt.xlabel(self.xlabel)
        if self.ylabel:
            plt.ylabel(self.ylabel)
        if self.xlim:
            plt.xlim(self.xlim)
        if self.ylim:
            plt.ylim(self.ylim)
        plt.xscale(self.xscale)
        plt.yscale(self.yscale)
        if self.legend:
            plt.legend()
        plt.show()


def grad_clipping(net, theta: float) -> None:
    """裁剪梯度范数，避免 RNN 梯度爆炸。"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def download(name: str, cache_dir: Path | str = DATA_DIR) -> str:
    """下载并缓存 DATA_HUB 中的文件。"""
    assert name in DATA_HUB, f"{name} 不存在于 DATA_HUB"
    url, sha1_hash = DATA_HUB[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / url.split("/")[-1]
    if fname.exists():
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data_chunk = f.read(1048576)
                if not data_chunk:
                    break
                sha1.update(data_chunk)
        if sha1.hexdigest() == sha1_hash:
            return str(fname)
    with urllib.request.urlopen(url, timeout=30) as response, open(fname, "wb") as f:
        while True:
            chunk = response.read(1048576)
            if not chunk:
                break
            f.write(chunk)
    return str(fname)


def download_extract(name: str, folder: str | None = None) -> str:
    """下载并解压 zip 文件。"""
    fname = Path(download(name))
    base_dir = fname.parent
    data_dir = fname.with_suffix("")
    if fname.suffix != ".zip":
        raise ValueError("当前轻量工具层只处理 zip 数据集")
    with zipfile.ZipFile(fname, "r") as zip_file:
        zip_file.extractall(base_dir)
    return str(base_dir / folder) if folder else str(data_dir)


def tokenize(lines: Sequence[str], token: str = "word") -> list[list[str]]:
    """把文本切成单词或字符 token。"""
    if token == "word":
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]
    raise ValueError(f"未知 token 类型: {token}")


class Vocab:
    """最小词表实现。"""

    def __init__(self, tokens=None, min_freq: int = 0, reserved_tokens=None):
        tokens = tokens or []
        reserved_tokens = reserved_tokens or []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ["<unk>"] + list(reserved_tokens)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        self.unk = self.token_to_idx["<unk>"]

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self[token] for token in tokens]
        return self.token_to_idx.get(tokens, self.unk)

    def to_tokens(self, indices):
        if isinstance(indices, (list, tuple)):
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[int(indices)]


def count_corpus(tokens) -> collections.Counter:
    if tokens and isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def truncate_pad(line, num_steps: int, padding_token: int):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def load_array(data_arrays, batch_size: int, is_train: bool = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, nn.Module):
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
    animator = Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        ylim=[0.3, 0.9],
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    return animator


def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.numpy()
        ax.imshow(img, cmap="gray")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def load_data_fashion_mnist(batch_size, resize=None, root: Path | str = DATA_DIR / "fashion-mnist"):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    root = str(root)
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0),
    )


DATA_HUB["time_machine"] = (
    DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)


def read_time_machine() -> list[str]:
    """返回时间机器数据集的文本行。"""
    fname = download("time_machine")
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens: int = -1):
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_sequential(corpus, batch_size: int, num_steps: int):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    def __init__(self, batch_size: int, num_steps: int, max_tokens: int = 10000):
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size = batch_size
        self.num_steps = num_steps

    def __iter__(self):
        return seq_data_iter_sequential(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size: int, num_steps: int, max_tokens: int = 10000):
    data_iter = SeqDataLoader(batch_size, num_steps, max_tokens)
    return data_iter, data_iter.vocab


class RNNModelScratch:
    """与教材一致的从零实现 RNN 包装器。"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn
        self.device = device

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


class RNNModel(nn.Module):
    """对 PyTorch RNN/GRU/LSTM 层做统一包装。"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        self.num_directions = 2 if self.rnn.bidirectional else 1
        self.linear = nn.Linear(self.num_hiddens * self.num_directions, vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        shape = (self.rnn.num_layers * self.num_directions, batch_size, self.num_hiddens)
        if isinstance(self.rnn, nn.LSTM):
            return (
                torch.zeros(shape, device=device),
                torch.zeros(shape, device=device),
            )
        return torch.zeros(shape, device=device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    for char in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[char])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return "".join(vocab.to_tokens(outputs))


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter=False):
    state, timer = None, Timer()
    metric = Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            elif isinstance(state, tuple):
                for s in state:
                    s.detach_()
        X, Y = X.to(device), Y.to(device)
        y = Y.T.reshape(-1)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    animator = Animator(xlabel="epoch", ylabel="perplexity", legend=["train"], xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f"perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {device}")
    print(predict_ch8("time traveller ", 50, net, vocab, device))
    return animator


def sgd(params, lr: float, batch_size: int) -> None:
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if device is None:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
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
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)

    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        legend=["train loss", "train acc", "test acc"],
    )
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3)
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
            if (i + 1) % max(1, num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {device}")
    return animator
