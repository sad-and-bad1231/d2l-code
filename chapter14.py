"""D2L 第 14 章代码整理版。

本文件聚焦 NLP 预训练主线，涵盖：
1. 词向量预训练（skip-gram + negative sampling）；
2. 子词分词（BPE）与预训练词向量读取；
3. BERT 相关数据处理、模型结构与最小训练入口。

说明：
- 默认入口只做轻量 shape / 数据流检查；
- 真实预训练任务都较慢，统一放到 `run_xxx()` / `train_xxx()` 中按需调用；
- 数据集下载失败时会打印跳过提示，不让脚本直接崩溃。
"""

from __future__ import annotations

import collections
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import chapter10 as ch10
import mini_d2l as d2l


# ==================== 1. PTB 数据集与词向量预训练 ====================
d2l.DATA_HUB["ptb"] = (
    d2l.DATA_URL + "ptb.zip",
    "319d85e578af0cdc590547f26231e4e31cdf1e42",
)


def read_ptb():
    """读取 PTB 训练语料。"""
    data_dir = Path(d2l.download_extract("ptb"))
    with open(data_dir / "ptb.train.txt", "r", encoding="utf-8") as f:
        return f.read().split("\n")


def subsample(sentences, vocab):
    """对子频极高的 token 做下采样。"""
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        return random.random() < math.sqrt(1e-4 / counter[token] * num_tokens)

    return [[token for token in line if keep(token)] for line in sentences], counter


def get_centers_and_contexts(corpus, max_window_size):
    """从语料中采样中心词与上下文。"""
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts


class RandomGenerator:
    """按离散概率分布做高效采样。"""

    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


def get_negatives(all_contexts, vocab, counter, K):
    """为每个上下文词列表采样负样本。"""
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def batchify(data):
    """把变长上下文/负样本打包成批量张量。"""
    max_len = max(len(context) + len(negative) for _, context, negative in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (
        torch.tensor(centers).reshape((-1, 1)),
        torch.tensor(contexts_negatives),
        torch.tensor(masks),
        torch.tensor(labels),
    )


class PTBDataset(torch.utils.data.Dataset):
    """PTB 词向量训练数据集。"""

    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return self.centers[index], self.contexts[index], self.negatives[index]

    def __len__(self):
        return len(self.centers)


def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """返回 PTB 预训练所需数据迭代器与词表。"""
    sentences = d2l.tokenize(read_ptb())
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)
    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)
    return data_iter, vocab


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """skip-gram 前向计算。"""
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class SigmoidBCELoss(nn.Module):
    """带 mask 的二元交叉熵。"""

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        )
        return out.mean(dim=1)


def train_word2vec(
    data_iter,
    vocab,
    embed_size=100,
    lr=0.002,
    num_epochs=5,
    device=None,
):
    """训练 skip-gram 词向量。"""
    if device is None:
        device = d2l.try_gpu()
    net = nn.Sequential(
        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
        nn.Embedding(num_embeddings=len(vocab), embedding_dim=embed_size),
    )
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = SigmoidBCELoss()

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for center, context_negative, mask, label in data_iter:
            center = center.to(device)
            context_negative = context_negative.to(device)
            mask = mask.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = loss(pred.reshape(label.shape).float(), label.float(), mask.float())
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
        print(f"epoch {epoch + 1}, loss {metric[0] / metric[1]:.4f}")
    return net


def get_similar_tokens(query_token, k, embed, vocab):
    """打印与某个词最相近的词。"""
    W = embed.weight.data
    x = W[vocab[query_token]]
    cos = torch.mv(W, x) / (
        torch.sqrt(torch.sum(W * W, dim=1) + 1e-9) * torch.sqrt((x * x).sum())
    )
    topk = torch.topk(cos, k=k + 1)[1].cpu().numpy().astype("int32")
    for i in topk[1:]:
        print(f"cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}")


def inspect_ptb_batch():
    """检查 PTB 批量张量形状。"""
    data_iter, vocab = load_data_ptb(batch_size=2, max_window_size=5, num_noise_words=5)
    for batch in data_iter:
        names = ["centers", "contexts_negatives", "masks", "labels"]
        for name, data in zip(names, batch):
            print(name, "shape:", data.shape)
        print("vocab size:", len(vocab))
        break


# ==================== 2. 子词与预训练词向量读取 ====================
def get_max_freq_pair(token_freqs):
    """找到当前频次最高的相邻符号对。"""
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return max(pairs, key=pairs.get)


def merge_symbols(max_freq_pair, token_freqs, symbols):
    """把最高频符号对合并为新符号。"""
    symbols.append("".join(max_freq_pair))
    new_token_freqs = {}
    bigram = " ".join(max_freq_pair)
    replacement = "".join(max_freq_pair)
    for token, freq in token_freqs.items():
        new_token = token.replace(bigram, replacement)
        new_token_freqs[new_token] = freq
    return new_token_freqs


def byte_pair_encoding(token_freqs, num_merges):
    """运行 BPE。"""
    symbols = ["<unk>"] + [chr(i) for i in range(ord("a"), ord("z") + 1)] + ["_"]
    token_freqs = {" ".join(list(token)) + " _": freq for token, freq in token_freqs.items()}
    for _ in range(num_merges):
        pair = get_max_freq_pair(token_freqs)
        token_freqs = merge_symbols(pair, token_freqs, symbols)
    return symbols, token_freqs


def segment_BPE(tokens, symbols):
    """用已有 BPE 符号表切分 token。"""
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        while start < len(token) and start < end:
            if token[start:end] in symbols:
                cur_output.append(token[start:end])
                start = end
                end = len(token)
            else:
                end -= 1
        outputs.append(cur_output if start == len(token) else ["<unk>"])
    return outputs


d2l.DATA_HUB["glove.6b.50d"] = (
    d2l.DATA_URL + "glove.6B.50d.zip",
    "0b8703943ccdb6eb788e6f091b8946e82231bc4d",
)


class TokenEmbedding:
    """读取预训练词向量。"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        data_dir = Path(d2l.download_extract(embedding_name))
        vec_file = data_dir / "vec.txt"
        idx_to_token = ["<unk>"]
        idx_to_vec = []
        with open(vec_file, "r", encoding="utf-8") as f:
            for line in f:
                elems = line.rstrip().split(" ")
                token, values = elems[0], elems[1:]
                if len(values) <= 1:
                    continue
                idx_to_token.append(token)
                idx_to_vec.append([float(v) for v in values])
        idx_to_vec = [[0.0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def inspect_bpe():
    """检查 BPE 合并与切分结果。"""
    token_freqs = {"fast_": 4, "faster_": 3, "tall_": 5, "taller_": 4}
    symbols, _ = byte_pair_encoding(token_freqs, num_merges=10)
    print("BPE symbols tail:", symbols[-10:])
    print("segment:", segment_BPE(["tallest_", "fatter_"], symbols))


# ==================== 3. WikiText-2 与 BERT 数据处理 ====================
d2l.DATA_HUB["wikitext-2"] = (
    d2l.DATA_URL + "wikitext-2.zip",
    "3c914d17d80b1459be871a5039ac23e752a53cbe",
)


def _read_wiki(data_dir):
    """读取 WikiText-2 并按段落切分。"""
    file_name = Path(data_dir) / "wiki.train.tokens"
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split(" . ") for line in lines if len(line.split(" . ")) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """构造 BERT 输入的 token 和 segment id。"""
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = "<mask>"
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = vocab.idx_to_token[random.randint(0, len(vocab) - 1)]
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ["<cls>", "<sep>"]:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
    )
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels = [], [], [], []
    for token_ids, pred_positions, mlm_pred_label_ids, segments, is_next in examples:
        all_token_ids.append(
            torch.tensor(token_ids + [vocab["<pad>"]] * (max_len - len(token_ids)), dtype=torch.long)
        )
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long)
        )
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)))
        )
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long)
        )
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (
        all_token_ids,
        all_segments,
        valid_lens,
        all_pred_positions,
        all_mlm_weights,
        all_mlm_labels,
        nsp_labels,
    )


class WikiTextDataset(torch.utils.data.Dataset):
    """BERT 预训练数据集。"""

    def __init__(self, paragraphs, max_len):
        paragraphs = [d2l.tokenize(paragraph, token="word") for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=["<pad>", "<mask>", "<cls>", "<sep>"])
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        examples = [
            (_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
            for tokens, segments, is_next in examples
        ]
        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels,
        ) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx],
        )

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """返回 WikiText-2 数据迭代器与词表。"""
    data_dir = d2l.download_extract("wikitext-2")
    paragraphs = _read_wiki(data_dir)
    train_set = WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab


# ==================== 4. BERT 模型 ====================
class BERTEncoder(nn.Module):
    """BERT 编码器。"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        key_size=768,
        query_size=768,
        value_size=768,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"{i}",
                ch10.EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    True,
                ),
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, : X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """遮蔽语言模型头。"""

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size),
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    """下一句预测头。"""

    def __init__(self, num_inputs, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)


class BERTModel(nn.Module):
    """完整 BERT 模型。"""

    def __init__(
        self,
        vocab_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
        key_size=768,
        query_size=768,
        value_size=768,
        hid_in_features=768,
        mlm_in_features=768,
        nsp_in_features=768,
    ):
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            num_layers,
            dropout,
            max_len=max_len,
            key_size=key_size,
            query_size=query_size,
            value_size=value_size,
        )
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X,
                         mlm_weights_X, mlm_Y, nsp_y):
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X)
    mlm_l = F.cross_entropy(
        mlm_Y_hat.reshape(-1, vocab_size),
        mlm_Y.reshape(-1),
        reduction="none",
    ) * mlm_weights_X.reshape(-1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    nsp_l = F.cross_entropy(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, vocab_size, devices, num_steps):
    """BERT 最小训练入口。"""
    net = nn.DataParallel(net, device_ids=[device.index for device in devices if device.type == "cuda"]) \
        if len([d for d in devices if d.type == "cuda"]) > 1 else net
    device = devices[0]
    net = net.to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    metric = d2l.Accumulator(4)

    while step < num_steps:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_x = valid_lens_x.to(device)
            pred_positions_X = pred_positions_X.to(device)
            mlm_weights_X = mlm_weights_X.to(device)
            mlm_Y = mlm_Y.to(device)
            nsp_y = nsp_y.to(device)
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, None, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X,
                mlm_weights_X, mlm_Y, nsp_y
            )
            l.backward()
            trainer.step()
            timer.stop()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            step += 1
            if step >= num_steps:
                break
    print(f"MLM loss {metric[0] / metric[3]:.3f}, NSP loss {metric[1] / metric[3]:.3f}")
    print(f"{metric[2] / timer.sum():.1f} sentence pairs/sec on {device}")
    return net


def get_bert_encoding(net, tokens_a, tokens_b=None, device=None, vocab=None):
    """获取一句或两句文本的 BERT 编码。"""
    if device is None:
        device = d2l.try_gpu()
    tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=device).unsqueeze(0)
    segments = torch.tensor(segments, device=device).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=device).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X


def inspect_bert_shapes():
    """检查 BERT 编码器、MLM 和 NSP 的输出形状。"""
    vocab_size, num_hiddens, num_layers, num_heads = 10000, 128, 2, 2
    net = BERTModel(
        vocab_size=vocab_size,
        num_hiddens=num_hiddens,
        norm_shape=[128],
        ffn_num_input=128,
        ffn_num_hiddens=256,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.2,
        key_size=128,
        query_size=128,
        value_size=128,
        hid_in_features=128,
        mlm_in_features=128,
        nsp_in_features=128,
    )
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0] * 8])
    valid_lens = torch.tensor([8, 6])
    pred_positions = torch.tensor([[1, 5], [2, 4]])
    encoded_X, mlm_Y_hat, nsp_Y_hat = net(tokens, segments, valid_lens, pred_positions)
    print("encoded shape:", encoded_X.shape)
    print("mlm output shape:", mlm_Y_hat.shape)
    print("nsp output shape:", nsp_Y_hat.shape)


def main():
    """默认只做轻量检查。"""
    try:
        inspect_ptb_batch()
    except Exception as err:
        print(f"跳过 PTB 数据检查：{err}")
        print("如需运行该部分，请确认当前环境可联网，并允许在项目目录下创建 data 缓存。")
    inspect_bpe()
    inspect_bert_shapes()

    # 以下任务都比较慢，按需取消注释。
    # data_iter, vocab = load_data_ptb(batch_size=512, max_window_size=5, num_noise_words=5)
    # net = train_word2vec(data_iter, vocab, embed_size=100, num_epochs=5)
    # get_similar_tokens("chip", 3, net[0], vocab)

    # train_iter, vocab = load_data_wiki(batch_size=64, max_len=64)
    # net = BERTModel(
    #     vocab_size=len(vocab),
    #     num_hiddens=128,
    #     norm_shape=[128],
    #     ffn_num_input=128,
    #     ffn_num_hiddens=256,
    #     num_heads=2,
    #     num_layers=2,
    #     dropout=0.2,
    #     key_size=128,
    #     query_size=128,
    #     value_size=128,
    #     hid_in_features=128,
    #     mlm_in_features=128,
    #     nsp_in_features=128,
    # )
    # train_bert(train_iter, net, len(vocab), [d2l.try_gpu()], num_steps=50)


if __name__ == "__main__":
    main()
