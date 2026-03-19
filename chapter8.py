# RNN
#
# 这是我跟写《动手学深度学习》RNN 章节时整理的代码。
# 文件现在按“数据准备 -> 从零实现 RNN -> PyTorch 简洁实现 -> 训练入口”组织，
# 方便复习时按模块阅读，也避免 import 这个文件时直接开始训练。

import collections
import math
import random
import re

import torch
from torch import nn
from torch.nn import functional as F

import mini_d2l as d2l


# -----------------------------
# 1. 读取《时光机器》数据集并构造词表
# -----------------------------
d2l.DATA_HUB["time_machine"] = (
    d2l.DATA_URL + "timemachine.txt",
    "090b5e7e70c295757f55df93cb0a180b9691891a",
)


def read_time_machine():
    """读取原始文本，并只保留英文字母。"""
    with open(d2l.download("time_machine"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    """把文本切分成单词或字符 token。"""
    if token == "word":
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]
    raise ValueError(f"unknown token type: {token}")


def count_corpus(tokens):
    """统计 token 频率。

    `tokens` 可能是:
    1. 一维列表: ['time', 'machine', ...]
    2. 二维列表: [['time', 'machine'], ['traveller', ...], ...]
    """
    if len(tokens) == 0:
        return collections.Counter()
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表。

    `token_to_idx` 用于 token -> 索引
    `idx_to_token` 用于索引 -> token
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 这里必须先展平 tokens 再统计，否则二维列表会因为元素是 list 而无法计数。
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # `<unk>` 表示词表外 token，固定使用索引 0。
        self.unk = 0
        uniq_tokens = ["<unk>"] + reserved_tokens
        uniq_tokens += [
            token for token, freq in self.token_freqs if freq >= min_freq
        ]

        self.idx_to_token = []
        self.token_to_idx = {}
        for token in uniq_tokens:
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def load_corpus_time_machine(max_tokens=-1):
    """将文本转成字符级语料索引序列。"""
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# -----------------------------
# 2. 序列数据迭代器
# -----------------------------
def seq_data_iter_random(corpus, batch_size, num_steps):
    """随机采样一个批量的子序列。"""
    corpus = corpus[random.randint(0, num_steps - 1) :]
    num_subseqs = (len(corpus) - 1) // num_steps

    # 每个子序列的起始索引。
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos : pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i : i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """顺序分区采样，适合保留相邻批次的状态。"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)

    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y


class SeqDataLoader:
    """基于本文件实现的数据加载器。"""

    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size = batch_size
        self.num_steps = num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(
    batch_size, num_steps, use_random_iter=False, max_tokens=10000
):
    """返回时光机器数据迭代器和词表。"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


# -----------------------------
# 3. 从零实现 RNN
# -----------------------------
def get_params(vocab_size, num_hiddens, device):
    """初始化 RNN 参数。"""
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 输入层到隐藏层、隐藏层到隐藏层、隐藏层到输出层。
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """初始化隐藏状态 H。"""
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    """RNN 前向传播。

    `inputs` 形状: (num_steps, batch_size, vocab_size)
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state
    outputs = []

    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)

    # 把每个时间步的输出沿 batch 维拼接，
    # 得到 (num_steps * batch_size, vocab_size)。
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """从零实现的 RNN 语言模型。"""

    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state = init_state
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        # X 原始形状: (batch_size, num_steps)
        # one-hot 后转置为: (num_steps, batch_size, vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """根据前缀生成后续字符。"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 先把 prefix 喂给模型，更新隐藏状态。
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 再开始自回归生成。
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return "".join(vocab.idx_to_token[i] for i in outputs)


def grad_clipping(net, theta):
    """梯度裁剪，防止梯度爆炸。"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params

    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练一个 epoch，返回困惑度和速度。"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)

    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 顺序采样时，需要保留状态；但要截断计算图，避免梯度无限回传。
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
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
    """训练字符级语言模型。"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel="epoch", ylabel="perplexity", legend=["train"], xlim=[10, num_epochs]
    )

    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)

    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)

    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter
        )
        if (epoch + 1) % 10 == 0:
            print(predict("time traveller"))
            animator.add(epoch + 1, [ppl])

    print(f"困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}")
    print(predict("time traveller"))
    print(predict("traveller"))


# -----------------------------
# 4. PyTorch 简洁实现
# -----------------------------
class RNNModel(nn.Module):
    """对 `nn.RNN` / `nn.GRU` / `nn.LSTM` 做一层统一封装。"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # inputs: (batch_size, num_steps)
        # one-hot + transpose -> (num_steps, batch_size, vocab_size)
        X = F.one_hot(inputs.T.long(), self.vocab_size).to(torch.float32)
        Y, state = self.rnn(X, state)

        # 把所有时间步展平后送入全连接层，输出每个位置的下一个字符概率。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        shape = (
            self.num_directions * self.rnn.num_layers,
            batch_size,
            self.num_hiddens,
        )
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros(shape, device=device)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


# -----------------------------
# 5. 示例入口
# -----------------------------
def run_scratch_rnn(num_hiddens=512, batch_size=32, num_steps=35, num_epochs=500, lr=1):
    """训练从零实现的 RNN。"""
    device = d2l.try_gpu()
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    net = RNNModelScratch(
        len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn
    )
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)


def run_scratch_rnn_random_iter(
    num_hiddens=512, batch_size=32, num_steps=35, num_epochs=500, lr=1
):
    """训练从零实现的 RNN，使用随机采样。"""
    device = d2l.try_gpu()
    train_iter, vocab = load_data_time_machine(
        batch_size, num_steps, use_random_iter=True
    )
    net = RNNModelScratch(
        len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn
    )
    train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=True)


def run_torch_rnn(num_hiddens=256, batch_size=32, num_steps=35, num_epochs=500, lr=1):
    """训练 PyTorch `nn.RNN` 版本。"""
    device = d2l.try_gpu()
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    net = RNNModel(rnn_layer, vocab_size=len(vocab)).to(device)

    print(predict_ch8("time traveller", 10, net, vocab, device))
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)


if __name__ == "__main__":
    # 建议按需取消注释，只运行你当前正在学习的部分。
    # run_scratch_rnn()
    # run_scratch_rnn_random_iter()
    # run_torch_rnn()
    print("chapter8.py 已切换为不依赖 d2l。请按需取消注释对应示例。")
