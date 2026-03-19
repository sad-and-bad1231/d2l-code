# d2l-code

一个不依赖 `d2l` 包的个人版 D2L 学习代码库。

目标很明确：
- 继续按照《动手学深度学习》的教学顺序学习
- 保留核心模型与训练思路
- 代码实现脱离 `d2l`
- 本地和 Colab / Kaggle 都能直接运行

目前已完成：
- `chapter3.py`
- `chapter4.py`
- `chapter5.py`
- `chapter6.py`
- `chapter7.py`
- `chapter8.py`
- `chapter9.py`
- 通用轻量工具层 `mini_d2l.py`
- Colab 模板 `colab_template.ipynb`

[Open In Colab](https://colab.research.google.com/github/sad-and-bad1231/d2l-code/blob/main/colab_template.ipynb)

## 项目结构

- [mini_d2l.py](mini_d2l.py)
  轻量工具层，替代原来常用的 `d2l` 功能，包括数据加载、训练循环、词表、设备选择等。

- [chapter3.py](chapter3.py) 到 [chapter9.py](chapter9.py)
  章节代码。默认入口不会直接开始长时间训练，需要你手动取消注释对应函数。

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
```

默认情况下，这些脚本只会打印提示，不会自动训练。  
要真正运行某个实验，请打开对应文件，把 `if __name__ == "__main__":` 下的目标函数取消注释。

## Colab 运行

最简单的方式：

1. 打开 `colab_template.ipynb`
2. 在 Colab 里切换到 GPU runtime
3. 运行安装和环境检查单元
4. 按章节取消注释要跑的实验

如果你想从 GitHub 克隆到 Colab，仓库地址就是：

```text
https://github.com/sad-and-bad1231/d2l-code.git
```

## 当前说明

- `chapter3/4/6/7` 里的 Fashion-MNIST 使用 `torchvision.datasets.FashionMNIST`
- `chapter8/9` 的文本数据会下载到仓库内 `data/` 目录
- 当前默认实现优先保证“教学流程清楚 + Colab 可运行”，不是工业级训练框架

## 下一步

后续可以继续补：
- attention / Transformer / BERT 相关章节
- 更完整的 README 示例图和训练结果
- 更细的工具层拆分，例如 `trainer.py`、`data.py`、`metrics.py`
