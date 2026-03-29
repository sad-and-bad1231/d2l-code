"""D2L 第 15 章代码整理版。

本文件聚焦 NLP 应用主线，涵盖：
1. 情感分析数据集与模型（BiRNN / TextCNN）；
2. 自然语言推断数据集与注意力模型；
3. BERT 在下游分类任务上的最小微调入口。

说明：
- 默认入口只做轻量 shape / 数据流检查；
- 真正训练任务统一放在 `train_xxx()` / `run_xxx()` 中按需调用；
- 数据集下载失败时会打印跳过提示，不让脚本直接崩溃。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import chapter14 as ch14
import mini_d2l as d2l


# ==================== 1. IMDb 情感分析 ====================
d2l.DATA_HUB["aclImdb"] = (
    d2l.DATA_URL + "aclImdb_v1.tar.gz",
    "01ada507287d82875905620988597833ad4e0903",
)


def read_imdb(data_dir, is_train):
    """读取 IMDb 原始文本与标签。"""
    data, labels = [], []
    split = "train" if is_train else "test"
    for label in ["pos", "neg"]:
        folder_name = Path(data_dir) / split / label
        for file in folder_name.iterdir():
            with open(file, "r", encoding="utf-8") as f:
                review = f.read().replace("\n", "")
                data.append(review)
                labels.append(1 if label == "pos" else 0)
    return data, labels


def tokenize_imdb(lines):
    return [re.sub(r"<br\s*/?>", " ", line.lower()).split() for line in lines]


def load_data_imdb(batch_size, num_steps=500):
    """返回 IMDb 情感分析数据迭代器与词表。"""
    data_dir = d2l.download_extract("aclImdb", folder="aclImdb")
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize_imdb(train_data[0])
    test_tokens = tokenize_imdb(test_data[0])
    vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=["<pad>"])
    train_features = torch.tensor(
        [d2l.truncate_pad(vocab[line], num_steps, vocab["<pad>"]) for line in train_tokens]
    )
    test_features = torch.tensor(
        [d2l.truncate_pad(vocab[line], num_steps, vocab["<pad>"]) for line in test_tokens]
    )
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])), batch_size, is_train=False)
    return train_iter, test_iter, vocab


class BiRNN(nn.Module):
    """双向 LSTM 情感分析模型。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(
            embed_size,
            num_hiddens,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        return self.decoder(encoding)


class TextCNN(nn.Module):
    """TextCNN 情感分析模型。"""

    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        embeddings = embeddings.permute(0, 2, 1)
        encoding = torch.cat(
            [torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self.convs],
            dim=1,
        )
        outputs = self.decoder(self.dropout(encoding))
        return outputs


def train_sentiment(net, train_iter, test_iter, lr=0.01, num_epochs=5, device=None):
    """训练情感分类模型。"""
    if device is None:
        device = d2l.try_gpu()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(
        xlabel="epoch",
        xlim=[1, num_epochs],
        legend=["train loss", "train acc", "test acc"],
    )
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(float(l) * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, device)
        animator.add(epoch + 1, (train_l, train_acc, test_acc))
        print(
            f"epoch {epoch + 1}: train loss {train_l:.4f}, "
            f"train acc {train_acc:.4f}, test acc {test_acc:.4f}"
        )
    return net


def predict_sentiment(net, vocab, sequence, device=None):
    """预测一条评论的情感极性。"""
    if device is None:
        device = d2l.try_gpu()
    sequence = torch.tensor(vocab[sequence.split()], device=device).reshape(1, -1)
    label = torch.argmax(net(sequence), dim=1)
    return "positive" if int(label) == 1 else "negative"


def inspect_sentiment_models():
    """检查 BiRNN 与 TextCNN 的输出形状。"""
    vocab_size = 100
    X = torch.ones((4, 10), dtype=torch.long)
    birnn = BiRNN(vocab_size, embed_size=16, num_hiddens=8, num_layers=2)
    textcnn = TextCNN(vocab_size, embed_size=16, kernel_sizes=[3, 4, 5], num_channels=[4, 4, 4])
    print("BiRNN output shape:", birnn(X).shape)
    print("TextCNN output shape:", textcnn(X).shape)


# ==================== 2. SNLI 自然语言推断 ====================
d2l.DATA_HUB["SNLI"] = (
    d2l.DATA_URL + "snli_1.0.zip",
    "9fcde07509c7e87ec61c640c1b2753d9041758e4",
)


def read_snli(data_dir, is_train):
    """读取 SNLI 前提句、假设句和标签。"""
    def extract_text(s):
        s = re.sub("\\(", "", s)
        s = re.sub("\\)", "", s)
        s = re.sub("\\s{2,}", " ", s)
        return s.strip()

    label_set = {"entailment": 0, "contradiction": 1, "neutral": 2}
    file_name = Path(data_dir) / "snli_1.0_train.txt" if is_train else Path(data_dir) / "snli_1.0_test.txt"
    with open(file_name, "r", encoding="utf-8") as f:
        rows = [row.split("\t") for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


class SNLIDataset(torch.utils.data.Dataset):
    """SNLI 数据集。"""

    def __init__(self, dataset, num_steps, vocab=None):
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(
                all_premise_tokens + all_hypothesis_tokens, min_freq=5, reserved_tokens=["<pad>"]
            )
        else:
            self.vocab = vocab
        self.premises = torch.tensor(
            [d2l.truncate_pad(self.vocab[line], num_steps, self.vocab["<pad>"]) for line in all_premise_tokens]
        )
        self.hypotheses = torch.tensor(
            [d2l.truncate_pad(self.vocab[line], num_steps, self.vocab["<pad>"]) for line in all_hypothesis_tokens]
        )
        self.labels = torch.tensor(dataset[2])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_data_snli(batch_size, num_steps=50):
    """返回 SNLI 数据迭代器与词表。"""
    data_dir = d2l.download_extract("SNLI")
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False)
    return train_iter, test_iter, train_set.vocab


class MLP(nn.Module):
    """Decomposable Attention 中使用的小 MLP。"""

    def __init__(self, num_inputs, num_hiddens, flatten, **kwargs):
        super().__init__(**kwargs)
        self.mlp = []
        self.mlp.append(nn.Dropout(0.2))
        self.mlp.append(nn.Linear(num_inputs, num_hiddens))
        self.mlp.append(nn.ReLU())
        if flatten:
            self.mlp.append(nn.Flatten(start_dim=1))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, X):
        return self.mlp(X)


class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.f = MLP(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        f_A = self.f(A)
        f_B = self.f(B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha


class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.g = MLP(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B


class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.h = MLP(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat


class DecomposableAttention(nn.Module):
    """自然语言推断的可分解注意力模型。"""

    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(embed_size, num_hiddens)
        self.compare = Compare(embed_size * 2, num_hiddens)
        self.aggregate = Aggregate(num_hiddens * 2, num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        return self.aggregate(V_A, V_B)


def train_nli(net, train_iter, test_iter, lr=0.001, num_epochs=5, device=None):
    """训练自然语言推断模型。"""
    if device is None:
        device = d2l.try_gpu()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for (premises, hypotheses), y in train_iter:
            optimizer.zero_grad()
            premises = premises.to(device)
            hypotheses = hypotheses.to(device)
            y = y.to(device)
            y_hat = net((premises, hypotheses))
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(float(l) * y.numel(), d2l.accuracy(y_hat, y), y.numel())
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        print(f"epoch {epoch + 1}: train loss {train_l:.4f}, train acc {train_acc:.4f}")
    return net


def predict_nli(net, vocab, premise, hypothesis, device=None):
    """预测一对句子的蕴含关系。"""
    if device is None:
        device = d2l.try_gpu()
    net.eval()
    premise_tokens = torch.tensor(vocab[premise.lower().split()], device=device).reshape(1, -1)
    hypothesis_tokens = torch.tensor(vocab[hypothesis.lower().split()], device=device).reshape(1, -1)
    label = torch.argmax(net((premise_tokens, hypothesis_tokens)), dim=1)
    return ["entailment", "contradiction", "neutral"][int(label)]


def inspect_nli_model():
    """检查可分解注意力模型输出形状。"""
    vocab = d2l.Vocab([["a", "b"], ["a", "c"]], min_freq=1)
    net = DecomposableAttention(vocab, embed_size=16, num_hiddens=32)
    premises = torch.ones((4, 10), dtype=torch.long)
    hypotheses = torch.ones((4, 10), dtype=torch.long)
    print("DecomposableAttention output shape:", net((premises, hypotheses)).shape)


# ==================== 3. BERT 下游分类 ====================
class BERTClassifier(nn.Module):
    """用 BERT 编码器做句对分类。"""

    def __init__(self, bert, **kwargs):
        super().__init__(**kwargs)
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(bert.hidden[0].out_features, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


def train_bert_classifier(net, train_iter, test_iter, lr=1e-4, num_epochs=5, device=None):
    """训练 BERT 分类器。"""
    if device is None:
        device = d2l.try_gpu()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for batch in train_iter:
            tokens_X, segments_X, valid_lens_x, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            y_hat = net((tokens_X, segments_X, valid_lens_x))
            l = loss(y_hat, labels)
            l.backward()
            optimizer.step()
            metric.add(float(l) * labels.numel(), d2l.accuracy(y_hat, labels), labels.numel())
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_metric = d2l.Accumulator(2)
        net.eval()
        with torch.no_grad():
            for batch in test_iter:
                tokens_X, segments_X, valid_lens_x, labels = [x.to(device) for x in batch]
                y_hat = net((tokens_X, segments_X, valid_lens_x))
                test_metric.add(d2l.accuracy(y_hat, labels), labels.numel())
        test_acc = test_metric[0] / test_metric[1]
        print(
            f"epoch {epoch + 1}: train loss {train_l:.4f}, "
            f"train acc {train_acc:.4f}, test acc {test_acc:.4f}"
        )
    return net


def inspect_bert_classifier():
    """检查 BERT 分类头输出形状。"""
    bert = ch14.BERTModel(
        vocab_size=1000,
        num_hiddens=64,
        norm_shape=[64],
        ffn_num_input=64,
        ffn_num_hiddens=128,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
        key_size=64,
        query_size=64,
        value_size=64,
        hid_in_features=64,
        mlm_in_features=64,
        nsp_in_features=64,
    )
    net = BERTClassifier(bert)
    tokens = torch.randint(0, 1000, (2, 8))
    segments = torch.zeros((2, 8), dtype=torch.long)
    valid_lens = torch.tensor([8, 6])
    print("BERTClassifier output shape:", net((tokens, segments, valid_lens)).shape)


def main():
    """默认只做轻量检查。"""
    inspect_sentiment_models()
    inspect_nli_model()
    inspect_bert_classifier()

    # 以下任务训练较慢，按需取消注释。
    # train_iter, test_iter, vocab = load_data_imdb(batch_size=64)
    # net = BiRNN(len(vocab), embed_size=100, num_hiddens=100, num_layers=2)
    # train_sentiment(net, train_iter, test_iter, lr=0.01, num_epochs=5)

    # train_iter, test_iter, vocab = load_data_snli(batch_size=128, num_steps=50)
    # net = DecomposableAttention(vocab, embed_size=100, num_hiddens=200)
    # train_nli(net, train_iter, test_iter, num_epochs=5)


if __name__ == "__main__":
    main()
