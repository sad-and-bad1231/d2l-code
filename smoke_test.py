"""整仓轻量检查脚本。

用途：
- 在 Colab 或本地环境中快速确认 chapter3-11 与 mini_d2l 可以正常导入；
- 只执行轻量级 shape 检查和小型演示，不启动长时间训练；
- 尽量避开需要联网下载的大数据集，只有显式启用时才检查相关部分。

推荐在真正开始 GPU 训练前先运行：

```bash
python smoke_test.py
```
"""

import chapter3
import chapter4
import chapter5
import chapter6
import chapter7
import chapter8
import chapter9
import chapter10
import chapter11
import mini_d2l as d2l


def inspect_chapter3_basic():
    """检查 Chapter 3 的基础数学与数据接口。"""
    import torch

    true_w = torch.tensor([2.0, -3.4])
    features, labels = chapter3.synthetic_data(true_w, 4.2, 8)
    w = torch.zeros((2, 1))
    b = torch.zeros(1)
    y_hat = chapter3.linreg(features, w, b)
    print("chapter3 synthetic_data shape:", features.shape, labels.shape)
    print("chapter3 linreg output shape:", y_hat.shape)
    print("chapter3 softmax shape:", chapter3.softmax(torch.randn(2, 3)).shape)


def inspect_chapter4_basic():
    """检查 Chapter 4 的基础函数与张量形状。"""
    import torch

    X = torch.tensor([[-1.0, 0.0, 2.0]])
    print("chapter4 relu output:", chapter4.relu(X))
    print("chapter4 dropout output shape:", chapter4.dropout_layer(torch.ones(2, 3), 0.5).shape)
    _, _, poly_features, labels, n_train, n_test = chapter4.build_polynomial_data(
        max_degree=4, n_train=4, n_test=4
    )
    print("chapter4 polynomial data shape:", poly_features.shape, labels.shape, n_train, n_test)


def inspect_chapter8_basic():
    """检查 Chapter 8 手写 RNN 的最小前向传播。"""
    import torch

    vocab_size, num_hiddens = 10, 16
    device = torch.device("cpu")
    params = chapter8.get_params(vocab_size, num_hiddens, device)
    state = chapter8.init_rnn_state(batch_size=2, num_hiddens=num_hiddens, device=device)
    inputs = torch.randn(5, 2, vocab_size)
    output, new_state = chapter8.rnn(inputs, state, params)
    print("chapter8 scratch rnn output shape:", output.shape)
    print("chapter8 scratch rnn state shape:", new_state[0].shape)


def inspect_chapter11_basic():
    """检查 Chapter 11 的优化器状态与调度器输出。"""
    momentum_states = chapter11.init_momentum_states(feature_dim=5)
    adam_states = chapter11.init_adam_states(feature_dim=5)
    factor_lrs, cosine_lrs = chapter11.demo_schedulers(num_steps=4)
    print("chapter11 momentum state shapes:", momentum_states[0].shape, momentum_states[1].shape)
    print("chapter11 adam state shapes:", adam_states[0][0].shape, adam_states[1][0].shape)
    print("chapter11 scheduler samples:", factor_lrs, cosine_lrs)


def run_basic_smoke_test(include_network_data=False):
    """运行项目级轻量检查。"""
    print("device =", d2l.try_gpu())
    print("modules imported successfully")

    print("[smoke] chapter3")
    inspect_chapter3_basic()

    print("[smoke] chapter4")
    inspect_chapter4_basic()

    print("[smoke] chapter5")
    chapter5.inspect_parameters()
    chapter5.demo_custom_layers()
    chapter5.demo_composition()
    chapter5.demo_gpu()

    print("[smoke] chapter6")
    chapter6.demo_corr2d()
    chapter6.demo_pool2d()
    chapter6.show_layer_shapes(chapter6.build_lenet())

    # Chapter 7: 这里不把所有 CNN 都过一遍。
    # smoke test 的目标是确认“现代 CNN 章节的结构定义与代表性模型前向传播没问题”，
    # 而不是把整章所有网络一次性全测完。这样能更稳，也更容易定位问题。
    print("[smoke] chapter7")
    chapter7.show_layer_shapes(chapter7.build_alexnet(), (1, 1, 224, 224))
    chapter7.show_layer_shapes(chapter7.build_lenet_with_batchnorm(), (1, 1, 28, 28))
    chapter7.show_layer_shapes(chapter7.build_resnet18(), (1, 1, 96, 96))

    print("[smoke] chapter8")
    inspect_chapter8_basic()

    print("[smoke] chapter9")
    chapter9.inspect_seq2seq_shapes()

    print("[smoke] chapter10")
    chapter10.inspect_attention_scoring()
    chapter10.inspect_seq2seq_attention_shapes()
    chapter10.inspect_multihead_attention_shapes()
    chapter10.inspect_transformer_shapes()

    print("[smoke] chapter11")
    inspect_chapter11_basic()

    if include_network_data:
        print("running optional network-data checks...")
        chapter9.inspect_nmt_batch()

    print("smoke test finished successfully")


if __name__ == "__main__":
    run_basic_smoke_test(include_network_data=False)
