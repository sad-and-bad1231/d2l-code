# d2l-code

不依赖 `d2l` 包的个人版 D2L 学习代码库。

目标很明确：
- 按照《动手学深度学习》的教学顺序学习
- 保留核心模型与训练思路
- 代码实现脱离 `d2l`包
- 本地和 Colab / Kaggle 都能直接运行

目前已完成：
- `chapter3.py`
- `chapter4.py`
- `chapter5.py`
- `chapter6.py`
- `chapter7.py`
- `chapter8.py`
- `chapter9.py`
- `chapter10.py`
- `chapter11.py`
- 通用轻量工具层 `mini_d2l.py`
- 轻量检查脚本 `smoke_test.py`
- Colab 模板 `colab_template.ipynb`

[Open In Colab](https://colab.research.google.com/github/sad-and-bad1231/d2l-code/blob/main/colab_template.ipynb)

## 项目结构

- [mini_d2l.py](mini_d2l.py)
  轻量工具层，替代原来常用的 `d2l` 功能，包括数据加载、训练循环、词表、设备选择等。

- [chapter3.py](chapter3.py) 到 [chapter11.py](chapter11.py)
  章节代码。默认入口不会直接开始长时间训练，而是打印本章可用函数，建议你按需手动调用。

- [smoke_test.py](smoke_test.py)
  整仓轻量检查脚本。适合在 Colab 开始正式训练前先跑一遍，确认模块导入、基础 shape 检查和跨章节依赖都没有问题。

- [colab_template.ipynb](colab_template.ipynb)
  用于在 Colab 上运行本仓库代码的模板 notebook。

- [requirements-colab.txt](requirements-colab.txt)
  Colab 最小依赖。

## 本地运行

建议使用你现有的 `conda` 环境，至少保证这些包可用：

```bash
pip install torch torchvision matplotlib numpy pandas
```

然后直接运行某一章：

```bash
python chapter3.py
python chapter6.py
python chapter8.py
python chapter11.py
python smoke_test.py
```

默认情况下，这些脚本只会打印可用入口，不会自动训练。  
真正运行实验时，推荐在 Python/Colab 中显式导入模块后调用目标函数，而不是修改脚本文件本身。

例如：

```python
import chapter9
import chapter10
import chapter11

chapter9.inspect_seq2seq_shapes()
chapter10.run_transformer_translation(num_epochs=50)
chapter11.demo_schedulers(num_steps=10)
```

## 跨章节依赖说明

- `chapter3` 到 `chapter7` 只依赖 `mini_d2l.py`
- `chapter8` 主要自包含，但也会使用 `mini_d2l.py` 的通用训练与下载工具
- `chapter9` 依赖 `mini_d2l.py`
- `chapter10` 依赖 `mini_d2l.py` 和 `chapter9.py`
- `chapter11` 依赖 `mini_d2l.py`

这意味着：
- 把整个仓库目录一起上传到 Colab 是安全的
- 只单独上传 `chapter10.py` 不够，因为它会导入 `chapter9.py`
- `chapter9/10` 第一次运行翻译相关实验时会下载数据到仓库内 `data/` 目录

## Colab 运行

最简单的方式：

1. 打开 `colab_template.ipynb`
2. 在 Colab 里切换到 GPU runtime
3. 运行安装和环境检查单元
4. 先运行 smoke test / 轻量检查单元
5. 再按章节逐个打开你要跑的实验

如果你想从 GitHub 克隆到 Colab，仓库地址就是：

```text
https://github.com/sad-and-bad1231/d2l-code.git
```

## 当前说明

- `chapter3/4/6/7` 里的 Fashion-MNIST 使用 `torchvision.datasets.FashionMNIST`
- `chapter8/9` 的文本数据会下载到仓库内 `data/` 目录
- `chapter10` 的翻译与 Transformer 相关实验依赖 `chapter9` 中的数据与 seq2seq 基础设施
- `chapter11` 的优化器实验默认使用空气动力学噪声数据集，并会下载到仓库内 `data/` 目录
- 当前默认实现优先保证“教学流程清楚 + Colab 可运行”，不是工业级训练框架

## 更新中......

后续可以继续补：
- attention / Transformer / BERT 相关章节
- 更完整的 README 示例图和训练结果
- 更细的工具层拆分，例如 `trainer.py`、`data.py`、`metrics.py`
