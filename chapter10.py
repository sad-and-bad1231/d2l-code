"""Chapter 10: 注意力机制与 Transformer。

本章按《动手学深度学习》的学习顺序整理，包含：
- 注意力汇聚的直观例子
- 加性注意力与缩放点积注意力
- 带注意力机制的 seq2seq 解码器
- 多头注意力
- 位置编码
- Transformer 编码器与解码器

设计目标：
- 不依赖第三方 `d2l` 包；
- 只依赖仓库内的 `mini_d2l.py` 与前一章已经整理好的 `chapter9.py`；
- 默认入口只做轻量 shape / 数据流检查，不直接启动长时间训练；
- 关键张量形状、模块职责、训练入口都补上中文注释，便于后续复习。
"""

import math

import torch
from torch import nn

import chapter9 as ch9
import mini_d2l as d2l


# ==================== 1. 注意力汇聚：核回归视角 ====================
def f(x):
    """教材中的目标函数，用于构造一维回归玩具数据。"""
    return 2 * torch.sin(x) + x**0.8


def generate_attention_pooling_data(n_train=50):
    """生成核回归示例所需的训练集与测试集。"""
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)
    return x_train, y_train, x_test, y_truth


def plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat):
    """可视化真实函数、预测曲线和训练样本点。"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4.5, 3))
    plt.plot(x_test.detach().cpu(), y_truth.detach().cpu(), label="Truth")
    plt.plot(x_test.detach().cpu(), y_hat.detach().cpu(), label="Pred")
    plt.scatter(x_train.detach().cpu(), y_train.detach().cpu(), alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([0, 5])
    plt.ylim([-1, 5])
    plt.legend()
    plt.tight_layout()
    plt.show()


def inspect_average_attention_pooling():
    """演示非参数注意力汇聚。

    这里把每个测试点都看成 query，
    把训练输入看成 keys，训练输出看成 values。
    """
    x_train, y_train, x_test, y_truth = generate_attention_pooling_data()
    n_train, n_test = len(x_train), len(x_test)

    # 每个测试输入都和所有训练输入做距离比较，因此要把 x_test 扩成二维。
    x_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    attention_weights = nn.functional.softmax(-(x_repeat - x_train) ** 2 / 2, dim=1)
    y_hat = torch.matmul(attention_weights, y_train)

    print("attention_weights shape:", attention_weights.shape)
    plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)
    d2l.show_heatmaps(
        attention_weights.unsqueeze(0).unsqueeze(0),
        xlabel="Sorted training inputs",
        ylabel="Sorted testing inputs",
        figsize=(4, 4),
    )


class NWKernelRegression(nn.Module):
    """Nadaraya-Watson 核回归。

    本质上是一个只有单个可学习缩放参数 `w` 的注意力模型：
    - query 与 key 距离越近，softmax 后权重越大；
    - 最终输出等于 values 的加权平均。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        self.attention_weights = None

    def forward(self, queries, keys, values):
        # queries: (num_queries,)
        # keys:    (num_queries, num_kv_pairs)
        # values:  (num_queries, num_kv_pairs)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2,
            dim=1,
        )
        return torch.bmm(
            self.attention_weights.unsqueeze(1), values.unsqueeze(-1)
        ).reshape(-1)


def train_nw_kernel_regression(num_epochs=5, lr=0.5):
    """训练带可学习带宽的核回归模型。"""
    x_train, y_train, x_test, y_truth = generate_attention_pooling_data()
    n_train = len(x_train)

    # leave-one-out 训练：
    # 每个训练样本预测自己时，不能把自己当作 key/value。
    x_tile = x_train.repeat((n_train, 1))
    y_tile = y_train.repeat((n_train, 1))
    mask = (1 - torch.eye(n_train)).type(torch.bool)
    keys = x_tile[mask].reshape((n_train, -1))
    values = y_tile[mask].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[1, num_epochs])

    for epoch in range(num_epochs):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f"epoch {epoch + 1}, loss {float(l.sum()):.6f}")
        animator.add(epoch + 1, float(l.sum()))

    # 测试时每个 query 都可以访问完整训练集。
    keys = x_train.repeat((len(x_test), 1))
    values = y_train.repeat((len(x_test), 1))
    y_hat = net(x_test, keys, values).detach()

    plot_kernel_reg(x_train, y_train, x_test, y_truth, y_hat)
    d2l.show_heatmaps(
        net.attention_weights.unsqueeze(0).unsqueeze(0),
        xlabel="Sorted training inputs",
        ylabel="Sorted testing inputs",
        figsize=(4, 4),
    )
    return net


# ==================== 2. 掩蔽 softmax 与两类基础注意力 ====================
def masked_softmax(X, valid_lens):
    """在最后一个轴上执行带 mask 的 softmax。

    常见场景：
    - 序列补齐后，末尾 `<pad>` 位置不应参与归一化；
    - 解码器自注意力中，未来时刻不应被当前时刻看到。
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)

    shape = X.shape
    if valid_lens.dim() == 1:
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        valid_lens = valid_lens.reshape(-1)

    X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """加性注意力。

    它先把 query 和 key 映射到同一隐藏空间，再用一个小前馈网络打分。
    当 query 维度与 key 维度不同，或想增强灵活性时，这种形式很常见。
    """

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens):
        queries = self.W_q(queries)
        keys = self.W_k(keys)

        # 广播后：
        # queries -> (batch_size, num_queries, 1, num_hiddens)
        # keys    -> (batch_size, 1, num_kv_pairs, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # scores: (batch_size, num_queries, num_kv_pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class DotProductAttention(nn.Module):
    """缩放点积注意力。

    它直接用 query 与 key 的点积打分，并除以 `sqrt(d)`，
    这是 Transformer 里使用的标准注意力形式。
    """

    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


def inspect_attention_scoring():
    """检查两种基础注意力的输出形状和注意力矩阵。"""
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    additive_attention = AdditiveAttention(
        key_size=2, query_size=20, num_hiddens=8, dropout=0.1
    )
    additive_attention.eval()
    additive_output = additive_attention(queries, keys, values, valid_lens)
    print("additive attention output shape:", additive_output.shape)

    d2l.show_heatmaps(
        additive_attention.attention_weights.reshape((1, 1, 2, 10)),
        xlabel="Keys",
        ylabel="Queries",
        figsize=(4, 3),
    )

    dot_queries = torch.normal(0, 1, (2, 1, 2))
    dot_attention = DotProductAttention(dropout=0.5)
    dot_attention.eval()
    dot_output = dot_attention(dot_queries, keys, values, valid_lens)
    print("dot-product attention output shape:", dot_output.shape)

    d2l.show_heatmaps(
        dot_attention.attention_weights.reshape((1, 1, 2, 10)),
        xlabel="Keys",
        ylabel="Queries",
        figsize=(4, 3),
    )


# ==================== 3. 带注意力的 seq2seq 解码器 ====================
class AttentionDecoder(ch9.Decoder):
    """带注意力机制解码器的统一接口。"""

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    """带 Bahdanau 注意力的 seq2seq 解码器。"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.attention = AdditiveAttention(
            key_size=num_hiddens,
            query_size=num_hiddens,
            num_hiddens=num_hiddens,
            dropout=dropout,
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens,
            num_hiddens,
            num_layers,
            dropout=dropout,
        )
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights = None

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # chapter9 中编码器输出：
        # output: (num_steps, batch_size, num_hiddens)
        # state:  (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state

        # X -> (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []

        for x in X:
            # 当前时间步的 query 取解码器最后一层隐状态。
            query = hidden_state[-1].unsqueeze(1)

            # context: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)

            # 把上下文向量与当前词嵌入拼接，再送入 GRU。
            x = torch.cat((context, x.unsqueeze(1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)

            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


def inspect_seq2seq_attention_shapes():
    """检查带注意力的编码器-解码器张量形状。"""
    encoder = ch9.Seq2SeqEncoder(
        vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2
    )
    decoder = Seq2SeqAttentionDecoder(
        vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2
    )
    encoder.eval()
    decoder.eval()

    X = torch.zeros((4, 7), dtype=torch.long)
    enc_outputs = encoder(X)
    state = decoder.init_state(enc_outputs, None)
    output, new_state = decoder(X, state)

    print("decoder output shape:", output.shape)
    print("state tuple length:", len(new_state))
    print("encoder memory shape:", new_state[0].shape)
    print("hidden state shape:", new_state[1].shape)


def collect_seq2seq_attention_weights(dec_attention_weight_seq):
    """把逐步解码得到的注意力权重整理成热图所需形状。

    `predict_seq2seq(..., save_attention_weights=True)` 返回的是“每一步保存一次”的列表，
    且每一步的解码器只解一个 token，因此列表里的每个元素内部只包含一个时间步的权重。
    """
    if not dec_attention_weight_seq:
        return None

    # step[0] 的形状通常是 (batch_size=1, 1, num_encoder_steps)
    weights = torch.cat([step[0] for step in dec_attention_weight_seq], dim=1)
    return weights.unsqueeze(0)


def run_seq2seq_attention_translation(
    embed_size=32,
    num_hiddens=32,
    num_layers=2,
    dropout=0.1,
    batch_size=64,
    num_steps=10,
    lr=0.005,
    num_epochs=250,
):
    """训练带加性注意力的 seq2seq 翻译模型。"""
    device = d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = ch9.load_data_nmt(batch_size, num_steps)

    encoder = ch9.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout
    )
    net = ch9.EncoderDecoder(encoder, decoder)

    ch9.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    last_attention = None

    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = ch9.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True
        )
        print(f"{eng} => {translation}, bleu {ch9.bleu(translation, fra, k=2):.3f}")
        last_attention = dec_attention_weight_seq

    heatmap_weights = collect_seq2seq_attention_weights(last_attention)
    if heatmap_weights is not None:
        d2l.show_heatmaps(
            heatmap_weights[:, :, :, : len(engs[-1].split()) + 1].cpu(),
            xlabel="Key positions",
            ylabel="Query positions",
            figsize=(5, 4),
        )

    return net, src_vocab, tgt_vocab


# ==================== 4. 多头注意力与位置编码 ====================
def transpose_qkv(X, num_heads):
    """为多头并行计算重排 Q / K / V 的形状。"""
    # 原始形状: (batch_size, num_steps, num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 变成: (batch_size, num_heads, num_steps, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)

    # 合并 batch 维与头数维，方便一次性做 batch matrix multiply。
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """把多头输出还原回单个张量的标准形状。"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力。"""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        num_heads,
        dropout,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionalEncoding(nn.Module):
    """正弦-余弦位置编码。"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # P 的形状: (1, max_len, num_hiddens)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens,
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


def inspect_multihead_attention_shapes():
    """检查多头注意力前向传播后的输出形状。"""
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(
        num_hiddens,
        num_hiddens,
        num_hiddens,
        num_hiddens,
        num_heads,
        0.5,
    )
    attention.eval()

    batch_size, num_queries = 2, 4
    num_kv_pairs = 6
    valid_lens = torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kv_pairs, num_hiddens))
    output = attention(X, Y, Y, valid_lens)
    print("multi-head attention output shape:", output.shape)


def inspect_positional_encoding():
    """绘制部分位置编码曲线，观察不同维度随位置变化的规律。"""
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()

    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, : X.shape[1], :]
    d2l.plot(
        torch.arange(num_steps),
        P[0, :, 6:10].T,
        xlabel="Row (position)",
        legend=[f"Col {int(d)}" for d in torch.arange(6, 10)],
        figsize=(6, 2.5),
    )


# ==================== 5. Transformer 编码器与解码器 ====================
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络。

    它对每个位置独立地做同一组 MLP 变换，
    不在时间步之间共享信息，时间步之间的信息交换主要由注意力完成。
    """

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接后接层规范化。"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer 编码器块。

    一个编码器块包含两层子结构：
    1. 多头自注意力
    2. 逐位置前馈网络
    每层子结构外面都包一层“残差 + LayerNorm”。
    """

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size,
            query_size,
            value_size,
            num_hiddens,
            num_heads,
            dropout,
            use_bias,
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(ch9.Encoder):
    """Transformer 编码器。"""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"block{i}",
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias,
                ),
            )
        self.attention_weights = None

    def forward(self, X, valid_lens, *args):
        # 乘以 sqrt(num_hiddens) 是为了让词嵌入的尺度与位置编码更匹配。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)

        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """Transformer 解码器中的第 i 个块。"""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        # state[2][self.i] 用于缓存当前解码层历史上已经生成过的表示：
        # - 训练时一次把整条目标序列都喂进去，因此起点是 None；
        # - 预测时逐 token 解码，需要不断拼接历史结果。
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device
            ).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    """Transformer 解码器。"""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                f"block{i}",
                DecoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
                ),
            )
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights = None

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]

        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def inspect_transformer_shapes():
    """检查 Transformer 编码器和解码器的输出形状。"""
    valid_lens = torch.tensor([3, 2])

    encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
    encoder_blk.eval()
    X = torch.ones((2, 100, 24))
    print("encoder block output shape:", encoder_blk(X, valid_lens).shape)

    encoder = TransformerEncoder(
        200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5
    )
    encoder.eval()
    enc_output = encoder(torch.ones((2, 100), dtype=torch.long), valid_lens)
    print("transformer encoder output shape:", enc_output.shape)

    decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
    decoder_blk.eval()
    state = [encoder_blk(X, valid_lens), valid_lens, [None]]
    print("decoder block output shape:", decoder_blk(X, state)[0].shape)


def collect_transformer_attention_weights(attention_weight_seq, num_layers, num_heads):
    """整理 Transformer 逐步解码时保存的注意力权重。

    返回：
    - 解码器自注意力权重，形状 `(num_layers, num_heads, num_queries, num_keys)`
    - 编码器-解码器注意力权重，形状相同
    """
    if not attention_weight_seq:
        return None, None

    grouped_weights = []
    for kind in range(2):
        layer_weights = []
        for layer_idx in range(num_layers):
            step_weights = [step[kind][layer_idx] for step in attention_weight_seq]
            layer_tensor = torch.cat(step_weights, dim=1)
            layer_weights.append(layer_tensor.reshape(1, num_heads, layer_tensor.shape[1], layer_tensor.shape[2]))
        grouped_weights.append(torch.cat(layer_weights, dim=0))
    return grouped_weights[0], grouped_weights[1]


def run_transformer_translation(
    num_hiddens=32,
    num_layers=2,
    dropout=0.1,
    batch_size=64,
    num_steps=10,
    lr=0.005,
    num_epochs=200,
    ffn_num_input=32,
    ffn_num_hiddens=64,
    num_heads=4,
    key_size=32,
    query_size=32,
    value_size=32,
):
    """训练 Transformer 机器翻译模型。"""
    device = d2l.try_gpu()
    norm_shape = [num_hiddens]
    train_iter, src_vocab, tgt_vocab = ch9.load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    decoder = TransformerDecoder(
        len(tgt_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    net = ch9.EncoderDecoder(encoder, decoder)

    ch9.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    last_translation = None
    last_attention = None

    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = ch9.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True
        )
        print(f"{eng} => {translation}, bleu {ch9.bleu(translation, fra, k=2):.3f}")
        last_translation = translation
        last_attention = dec_attention_weight_seq

    # 这里的编码器注意力来自最后一次预测，因此 batch_size=1。
    enc_attention_weights = torch.cat(net.encoder.attention_weights, dim=0).reshape(
        num_layers, num_heads, -1, num_steps
    )
    d2l.show_heatmaps(
        enc_attention_weights.cpu(),
        xlabel="Key positions",
        ylabel="Query positions",
        titles=[f"Head {i}" for i in range(1, num_heads + 1)],
        figsize=(7, 3.5),
    )

    dec_self_attention, dec_inter_attention = collect_transformer_attention_weights(
        last_attention, num_layers, num_heads
    )
    if dec_self_attention is not None:
        d2l.show_heatmaps(
            dec_self_attention[:, :, :, : len(last_translation.split()) + 1].cpu(),
            xlabel="Key positions",
            ylabel="Query positions",
            titles=[f"Head {i}" for i in range(1, num_heads + 1)],
            figsize=(7, 3.5),
        )
        d2l.show_heatmaps(
            dec_inter_attention.cpu(),
            xlabel="Key positions",
            ylabel="Query positions",
            titles=[f"Head {i}" for i in range(1, num_heads + 1)],
            figsize=(7, 3.5),
        )

    return net, src_vocab, tgt_vocab


def main():
    """默认只做轻量检查，不直接开始长时间训练。"""
    inspect_attention_scoring()
    inspect_seq2seq_attention_shapes()
    inspect_multihead_attention_shapes()
    inspect_transformer_shapes()

    print("\n可按需手动运行以下训练入口：")
    print("- inspect_average_attention_pooling()")
    print("- train_nw_kernel_regression()")
    print("- inspect_positional_encoding()")
    print("- run_seq2seq_attention_translation()")
    print("- run_transformer_translation()")


if __name__ == "__main__":
    main()
