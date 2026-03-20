"""Chapter 9: 现代循环神经网络。

- GRU
- LSTM
- 机器翻译与数据处理
- 编码器-解码器
- seq2seq
- others

"""
import collections
import math
import os

import torch
from torch import nn

import mini_d2l as d2l


# ==================== 1. 从零实现 GRU ====================
def get_gru_params(vocab_size, num_hiddens, device):
    """初始化 GRU 的全部可学习参数。

    GRU 在每个时间步需要三组门控相关参数：
    - 更新门 Z决定“保留多少旧隐状态”
    - 重置门 R决定“计算候选隐状态时看多少旧信息”
    - 候选隐状态 H_tilda基于当前输入和部分旧状态生成新内容

    额外还有输出层参数，用于把隐状态映射到词表大小的 logits。
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 使用较小随机值初始化，避免初期激活过大。
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        # 返回一组“输入权重、隐状态权重、偏置”。
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device),
        )

    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐状态参数

    # 输出层：H -> vocab logits
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [
        W_xz, W_hz, b_z,
        W_xr, W_hr, b_r,
        W_xh, W_hh, b_h,
        W_hq, b_q,
    ]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    """初始化 GRU 隐状态 H。"""
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    """按时间步手写 GRU 前向传播。

    参数约定：
    - inputs 形状：(num_steps, batch_size, vocab_size)
      这里输入通常是 one-hot 表示
    - state 是一个元组，内部只有当前隐状态 H
    - 返回：
      - 所有时间步拼接后的输出，形状：(num_steps * batch_size, vocab_size)
      - 最后一个时间步的隐状态
    """
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    (H,) = state
    outputs = []

    for X in inputs:
        # 1) 更新门：决定保留多少旧隐状态 H
        Z = torch.sigmoid(X @ W_xz + H @ W_hz + b_z)

        # 2) 重置门：决定计算候选状态时，旧隐状态参与多少
        R = torch.sigmoid(X @ W_xr + H @ W_hr + b_r)

        # 3) 候选隐状态：当前输入 + 经过 R 筛选的旧隐状态
        H_tilda = torch.tanh(X @ W_xh + (R * H) @ W_hh + b_h)

        # 4) 新隐状态：在“旧状态”和“候选状态”之间做加权融合
        H = Z * H + (1 - Z) * H_tilda

        # 5) 输出层：用当前隐状态预测下一个词元
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H,)


# ==================== 2. 从零实现 LSTM ====================
def get_lstm_params(vocab_size, num_hiddens, device):
    """初始化 LSTM 的全部参数。

    LSTM 相比普通 RNN 多了记忆元 C，并且有 4 组参数：
    - 输入门 I：当前候选信息写入多少
    - 遗忘门 F：旧记忆保留多少
    - 输出门 O：当前记忆暴露给隐状态多少
    - 候选记忆 C_tilda：当前时刻准备写入的新内容
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (
            normal((num_inputs, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device),
        )

    W_xi, W_hi, b_i = three()  # 输入门
    W_xf, W_hf, b_f = three()  # 遗忘门
    W_xo, W_ho, b_o = three()  # 输出门
    W_xc, W_hc, b_c = three()  # 候选记忆

    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q,
    ]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    """初始化 LSTM 的隐状态 H 和记忆元 C。"""
    return (
        torch.zeros((batch_size, num_hiddens), device=device),
        torch.zeros((batch_size, num_hiddens), device=device),
    )


def lstm(inputs, state, params):
    """按时间步手写 LSTM 前向传播。"""
    [
        W_xi, W_hi, b_i,
        W_xf, W_hf, b_f,
        W_xo, W_ho, b_o,
        W_xc, W_hc, b_c,
        W_hq, b_q,
    ] = params
    H, C = state
    outputs = []

    for X in inputs:
        # 1) 三个门和候选记忆
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_tilda = torch.tanh(X @ W_xc + H @ W_hc + b_c)

        # 2) 先更新长期记忆 C，再从 C 产生短期输出 H
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)

        # 3) 由当前隐状态映射到词表预测
        Y = H @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)


# ==================== 3. 机器翻译数据处理 ====================
d2l.DATA_HUB["fra-eng"] = (
    d2l.DATA_URL + "fra-eng.zip",
    "94646ad1522d915e7b0f9296181140edcf86a4f5",
)


def read_data_nmt():
    """下载并读取英法翻译数据集原始文本。"""
    data_dir = d2l.download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra.txt"), "r", encoding="utf-8") as f:
        return f.read()


def preprocess_nmt(text):
    """对英法数据集做最基础清洗。

    处理内容：
    - 把不间断空格替换成普通空格
    - 全部转成小写
    - 在标点前补空格，让标点也能被单独切成 token
    """

    def no_space(char, prev_char):
        return char in set(",.!?") and prev_char != " "

    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    out = [
        " " + char if i > 0 and no_space(char, text[i - 1]) else char
        for i, char in enumerate(text)
    ]
    return "".join(out)


def tokenize_nmt(text, num_examples=None):
    """把英法平行语料切成 token 序列。

    返回：
    - source: 英文 token 列表的列表
    - target: 法文 token 列表的列表
    """
    source, target = [], []
    for i, line in enumerate(text.split("\n")):
        # `>=` 更符合“只取前 num_examples 条样本”的直觉。
        if num_examples is not None and i >= num_examples:
            break
        parts = line.split("\t")
        if len(parts) == 2:
            source.append(parts[0].split(" "))
            target.append(parts[1].split(" "))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """把序列统一成固定长度。

    - 太长：截断
    - 太短：在右侧补 `<pad>`
    """
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """把 token 序列转成张量，并计算有效长度。

    注意：
    - 每条样本末尾都会追加 `<eos>`，表示句子结束
    - 有效长度 valid_len 用于后续 mask，避免把 `<pad>` 也算进损失
    """
    lines = [vocab[line] for line in lines]
    lines = [line + [vocab["<eos>"]] for line in lines]
    array = torch.tensor(
        [truncate_pad(line, num_steps, vocab["<pad>"]) for line in lines]
    )
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """构造机器翻译数据迭代器和源/目标词表。"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)

    src_vocab = d2l.Vocab(
        source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )
    tgt_vocab = d2l.Vocab(
        target, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"]
    )

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# ==================== 4. 编码器 - 解码器接口 ====================
class Encoder(nn.Module):
    """编码器接口。

    这里定义的不是具体网络，而是统一接口：
    任何编码器只要实现 `forward`，都可以塞进 EncoderDecoder 框架。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """解码器接口。

    解码器除了 `forward` 外，还要能根据编码器输出初始化自身状态。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """标准 seq2seq 外壳。

    调用流程：
    1. 编码器读取源句子
    2. 解码器根据编码结果初始化状态
    3. 解码器读取目标端输入并逐步输出预测
    """

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


# ==================== 5. Seq2Seq 编码器与解码器 ====================
class Seq2SeqEncoder(Encoder):
    """基于 GRU 的 seq2seq 编码器。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        # 词嵌入：把离散 token id 变成稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size, num_hiddens, num_layers, dropout=dropout
        )

    def forward(self, X, *args):
        # X: (batch_size, num_steps)
        X = self.embedding(X)
        # GRU 默认要求时间步在第 0 维，因此要转成
        # (num_steps, batch_size, embed_size)
        X = X.permute(1, 0, 2)

        # output: 每个时间步顶层隐藏状态
        # state: 最后一个时间步、每一层的隐藏状态
        output, state = self.rnn(X)
        return output, state


class Seq2SeqDecoder(Decoder):
    """不带注意力机制的 seq2seq 解码器。

    核心思想：
    - 编码器最后一层最后时刻的隐状态，视为源句子的“上下文向量”
    - 解码器每个时间步都把“当前输入词嵌入”和“上下文向量”拼接起来
    """

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens,
            num_hiddens,
            num_layers,
            dropout=dropout,
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 这里直接把编码器最终隐状态传给解码器。
        return enc_outputs[1]

    def forward(self, X, state):
        # X: (batch_size, num_steps)
        X = self.embedding(X).permute(1, 0, 2)

        # state[-1] 是“最后一层”的隐藏状态，形状为
        # (batch_size, num_hiddens)
        # 解码时把它复制到每个时间步，作为固定上下文。
        context = state[-1].repeat(X.shape[0], 1, 1)

        # 把当前词嵌入和上下文拼起来再送入 GRU。
        X_and_context = torch.cat((X, context), dim=2)
        output, state = self.rnn(X_and_context, state)

        # 线性层把隐藏状态映射回词表维度。
        output = self.dense(output).permute(1, 0, 2)
        return output, state


# ==================== 6. Mask 与损失函数 ====================
def sequence_mask(X, valid_len, value=0):
    """把每个样本中超出有效长度的位置屏蔽掉。

    例子：
    - 如果某条序列 valid_len=3
    - 那么第 4 个位置及以后都不应该参与损失计算
    """
    maxlen = X.size(1)
    mask = (
        torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """适用于变长序列的交叉熵损失。

    普通交叉熵会把 `<pad>` 位置也一起算进去；
    这里通过 mask 把这些位置的权重设为 0。
    """

    def forward(self, pred, label, valid_len):
        # pred:  (batch_size, num_steps, vocab_size)
        # label: (batch_size, num_steps)
        # valid_len: (batch_size,)
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)

        # 需要让 CrossEntropyLoss 输出每个位置的独立损失。
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)

        # 对无效位置乘 0，再沿时间步求平均。
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# ==================== 7. 训练、预测与评估 ====================
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练 seq2seq 模型。"""

    def xavier_init_weights(module):
        # Xavier 初始化有助于保持前后向方差稳定。
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.GRU):
            for param_name in module._flat_weights_names:
                if "weight" in param_name:
                    nn.init.xavier_uniform_(module._parameters[param_name])

    net.apply(xavier_init_weights)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 损失总和、目标词元总数

        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]

            # teacher forcing：
            # 解码器输入是 `<bos> + 目标序列去掉最后一个词`
            bos = torch.tensor(
                [tgt_vocab["<bos>"]] * Y.shape[0], device=device
            ).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)

            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()

            # RNN 类模型常见操作：梯度裁剪，避免梯度爆炸。
            d2l.grad_clipping(net, 1)
            optimizer.step()

            num_tokens = Y_valid_len.sum()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))

    print(
        f"loss {metric[0] / metric[1]:.3f}, "
        f"{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}"
    )


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """使用训练好的 seq2seq 模型做贪心解码。"""
    net.eval()

    # 1) 源句子分词、转 id、补上 `<eos>`
    src_tokens = src_vocab[src_sentence.lower().split(" ")] + [src_vocab["<eos>"]]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])

    # 2) 编码器输入需要带 batch 维
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0
    )
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)

    # 3) 解码从 `<bos>` 开始
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device),
        dim=0,
    )

    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)

        # 贪心解码：每步只取当前概率最大的词
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        # 本章这个解码器没有 attention_weights，
        # 这里保留接口是为了和后续章节兼容。
        if save_attention_weights and hasattr(net.decoder, "attention_weights"):
            attention_weight_seq.append(net.decoder.attention_weights)

        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)

    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """计算 BLEU。

    这里只实现书中的基础版本：
    - 先加上长度惩罚项
    - 再累计 1-gram 到 k-gram 的匹配情况
    """
    pred_tokens = pred_seq.split(" ") if pred_seq else []
    label_tokens = label_seq.split(" ")

    len_pred, len_label = len(pred_tokens), len(label_tokens)
    if len_pred == 0:
        return 0.0

    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        label_subs = collections.defaultdict(int)

        for i in range(len_label - n + 1):
            label_subs[" ".join(label_tokens[i: i + n])] += 1

        for i in range(len_pred - n + 1):
            ngram = " ".join(pred_tokens[i: i + n])
            if label_subs[ngram] > 0:
                num_matches += 1
                label_subs[ngram] -= 1

        denom = len_pred - n + 1
        if denom <= 0 or num_matches == 0:
            return 0.0
        score *= math.pow(num_matches / denom, math.pow(0.5, n))

    return score


# ==================== 8. 训练与演示入口 ====================
def run_gru_scratch():
    """运行从零实现的 GRU 语言模型训练。"""
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens, num_epochs, lr = 256, 500, 1
    device = d2l.try_gpu()

    model = d2l.RNNModelScratch(
        len(vocab), num_hiddens, device, get_gru_params, init_gru_state, gru
    )
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


def run_lstm_scratch():
    """运行从零实现的 LSTM 语言模型训练。"""
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens, num_epochs, lr = 256, 500, 1
    device = d2l.try_gpu()

    model = d2l.RNNModelScratch(
        len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm
    )
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


def run_multilayer_lstm():
    """运行多层 LSTM 的简洁实现。"""
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    device = d2l.try_gpu()

    lstm_layer = nn.LSTM(vocab_size, num_hiddens, num_layers)
    model = d2l.RNNModel(lstm_layer, len(vocab)).to(device)

    num_epochs, lr = 500, 2
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


def inspect_nmt_batch():
    """打印一个机器翻译 batch，帮助理解数据张量长什么样。"""
    train_iter, src_vocab, _ = load_data_nmt(batch_size=2, num_steps=8)
    raw_text = preprocess_nmt(read_data_nmt())
    source, _ = tokenize_nmt(raw_text)

    print("truncate_pad 示例：")
    print(truncate_pad(src_vocab[source[0]], 10, src_vocab["<pad>"]))

    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print("X:", X.type(torch.int32))
        print("X 的有效长度:", X_valid_len)
        print("Y:", Y.type(torch.int32))
        print("Y 的有效长度:", Y_valid_len)
        break


def inspect_seq2seq_shapes():
    """打印编码器/解码器的输出形状，帮助对齐张量维度。"""
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print("encoder output shape:", output.shape)
    print("encoder state shape:", state.shape)

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()
    dec_state = decoder.init_state((output, state))
    dec_output, dec_state = decoder(X, dec_state)
    print("decoder output shape:", dec_output.shape)
    print("decoder state shape:", dec_state.shape)


def run_seq2seq_translation():
    """训练基础 seq2seq 翻译模型，并输出几个示例翻译。"""
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 300
    device = d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    net = EncoderDecoder(encoder, decoder)

    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, _ = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device
        )
        print(f"{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}")


def main():
    """默认只打印可选入口。

    本章里既有语言模型长训练，也有机器翻译数据下载。
    因此直接运行脚本时不自动开跑，只给出可选函数列表。
    """
    print("Chapter 9 可用入口：")
    print("- inspect_seq2seq_shapes()")
    print("- inspect_nmt_batch()")
    print("- run_gru_scratch()")
    print("- run_lstm_scratch()")
    print("- run_multilayer_lstm()")
    print("- run_seq2seq_translation()")
    print("注意：NMT 相关函数第一次运行会联网下载 `fra-eng` 数据集。")


if __name__ == "__main__":
    main()
