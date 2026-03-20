"""整仓轻量检查脚本。

用途：
- 在 Colab 或本地环境中快速确认 chapter3-10 与 mini_d2l 可以正常导入；
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
import mini_d2l as d2l


def run_basic_smoke_test(include_network_data=False):
    """运行项目级轻量检查。"""
    print("device =", d2l.try_gpu())
    print("modules imported successfully")

    # Chapter 3-6: 只跑不依赖外部下载的小型检查。
    chapter5.inspect_parameters()
    chapter5.demo_custom_layers()
    chapter5.demo_composition()
    chapter5.demo_gpu()

    chapter6.demo_corr2d()
    chapter6.demo_pool2d()
    chapter6.show_layer_shapes(chapter6.build_lenet())

    # Chapter 7: 只检查网络结构和输出形状，不训练。
    chapter7.show_layer_shapes(chapter7.build_alexnet(), (1, 1, 224, 224))
    chapter7.show_layer_shapes(chapter7.build_vgg(chapter7.small_vgg_arch()), (1, 1, 224, 224))
    chapter7.show_layer_shapes(chapter7.build_nin(), (1, 1, 224, 224))
    chapter7.show_layer_shapes(chapter7.build_googlenet(), (1, 1, 96, 96))
    chapter7.show_layer_shapes(chapter7.build_lenet_with_batchnorm(), (1, 1, 28, 28))
    chapter7.show_layer_shapes(chapter7.build_resnet18(), (1, 1, 96, 96))
    chapter7.show_layer_shapes(chapter7.build_densenet(), (1, 1, 96, 96))

    # Chapter 8-10: 仅跑张量形状与注意力结构检查。
    chapter9.inspect_seq2seq_shapes()
    chapter10.inspect_attention_scoring()
    chapter10.inspect_seq2seq_attention_shapes()
    chapter10.inspect_multihead_attention_shapes()
    chapter10.inspect_transformer_shapes()

    if include_network_data:
        print("running optional network-data checks...")
        chapter9.inspect_nmt_batch()

    print("smoke test finished successfully")


if __name__ == "__main__":
    run_basic_smoke_test(include_network_data=False)
