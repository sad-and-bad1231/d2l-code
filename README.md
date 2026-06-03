# d2l-code

不依赖第三方 `d2l` 包的个人版《动手学深度学习》学习代码库。

这个仓库的目标是保留 D2L 的核心教学顺序、模型思路和可运行示例，同时把常用工具函数沉淀到仓库内的 `mini_d2l.py`，方便在本地、Colab 和 Kaggle 中复用。

[Open In Colab](https://colab.research.google.com/github/sad-and-bad1231/d2l-code/blob/main/colab_template.ipynb)

## 项目定位

- 按照《动手学深度学习》的章节顺序学习。
- 章节代码保持扁平结构，便于直接打开、阅读和运行。
- 默认脚本入口只打印可用函数或执行轻量检查，不自动启动长时间训练。
- 数据集按需下载到 `data/`，不会提交到 Git。
- 当前优先保证教学流程清楚和实验可复现，不是工业级训练框架。

## 当前内容

- `chapter3.py` 到 `chapter15.py`
- 通用轻量工具层 `mini_d2l.py`
- Kaggle 房价预测示例 `houseprice.py`
- 轻量检查脚本 `smoke_test.py`
- Colab 模板 `colab_template.ipynb`
- pytest 基线测试 `tests/`
- CI、依赖、贡献说明和 MIT 许可证

## 目录结构

```text
.
├── chapter3.py ... chapter15.py   # 按章节整理的学习代码
├── mini_d2l.py                    # 仓库内轻量工具层
├── houseprice.py                  # Kaggle 房价预测示例
├── smoke_test.py                  # 手动轻量检查入口
├── tests/                         # pytest 静态与轻量测试
├── colab_template.ipynb           # Colab 运行模板
├── requirements.txt               # 运行依赖
├── requirements-dev.txt           # 开发与 CI 依赖
└── pyproject.toml                 # Python 工程配置
```

## 环境要求

推荐使用 Python 3.10-3.12。当前不把 Python 3.13 作为正式支持目标，因为 PyTorch 与部分教学环境的兼容性更稳妥地落在 3.10-3.12。

安装运行依赖：

```bash
pip install -r requirements.txt
```

安装开发依赖：

```bash
pip install -r requirements-dev.txt
```

Colab 环境也可以继续使用：

```bash
pip install -r requirements-colab.txt
```

## 快速检查

轻量测试：

```bash
pytest
```

语法检查：

```bash
python -m compileall -q .
```

Lint：

```bash
ruff check .
```

手动 smoke test：

```bash
python smoke_test.py
```

`smoke_test.py` 会导入多个章节并做小型 shape 检查，需要当前环境已安装 `torch` 和 `torchvision`。它不会默认运行长时间训练，也不会默认下载大型数据集。

## 运行示例

直接运行某一章：

```bash
python chapter3.py
python chapter6.py
python chapter8.py
python chapter11.py
python chapter12.py
python chapter13.py
python chapter14.py
python chapter15.py
```

推荐在 Python、Notebook 或 Colab 中按需导入后调用目标函数：

```python
import chapter9
import chapter10
import chapter11
import chapter12
import chapter13
import chapter14
import chapter15

chapter9.inspect_seq2seq_shapes()
chapter10.inspect_transformer_shapes()
chapter11.demo_schedulers(num_steps=10)
chapter12.inspect_hardware()
chapter13.inspect_anchor_shapes()
chapter14.inspect_bert_shapes()
chapter15.inspect_nli_model()
```

## 跨章节依赖

- `chapter3` 到 `chapter7` 只依赖 `mini_d2l.py`。
- `chapter8` 使用 `mini_d2l.py` 的通用训练与下载工具。
- `chapter9` 依赖 `mini_d2l.py`。
- `chapter10` 依赖 `mini_d2l.py` 和 `chapter9.py`。
- `chapter11` 依赖 `mini_d2l.py`。
- `chapter12` 依赖 `mini_d2l.py` 和 `chapter7.py`。
- `chapter13` 依赖 `mini_d2l.py`、`chapter7.py` 与 `torchvision`。
- `chapter14` 依赖 `mini_d2l.py` 和 `chapter10.py`。
- `chapter15` 依赖 `mini_d2l.py`、`chapter14.py`。
- `houseprice.py` 依赖 `mini_d2l.py`，不再依赖第三方 `d2l` 包。

## 数据与下载

- Fashion-MNIST、Time Machine、NMT、PTB、WikiText-2、IMDb、SNLI 等数据会在对应函数首次运行时按需下载。
- 下载缓存目录默认为仓库内 `data/`，该目录已被 `.gitignore` 忽略。
- Kaggle 相关示例需要你提供对应比赛数据或手动运行相关入口。
- CI 和默认 pytest 不下载大数据集，不运行长时间训练。

## Colab 运行

最简单的方式：

1. 打开 `colab_template.ipynb`。
2. 在 Colab 里切换到 GPU runtime。
3. 从上到下运行安装、环境检查、smoke test 和章节示例单元。
4. 按需修改最后的章节实验单元，不建议直接修改章节脚本文件。

也可以从 GitHub 克隆：

```text
https://github.com/sad-and-bad1231/d2l-code.git
```

## 常见问题

**为什么不依赖第三方 `d2l` 包？**

为了让学习代码更透明，也方便在不同环境中直接复制、调试和修改。常用工具函数集中在 `mini_d2l.py`。

**为什么不把代码拆成标准 Python 包？**

这个仓库的主要目标是按章节学习。扁平结构牺牲了一些工程分层，但更容易对照教材阅读。

**为什么 CI 不运行完整训练？**

完整训练耗时长、依赖 GPU 或联网数据。CI 只验证语法、基础工程配置和轻量测试，训练实验建议在本地或 Colab 手动运行。

## 后续方向

- 更完整的 README 示例图和训练结果。
- 更细的工具层拆分，例如 `trainer.py`、`data.py`、`metrics.py`。
- 更多章节级轻量检查和 Colab 可复现实验单元。

## 贡献

欢迎通过 PR 改进文档、轻量测试、章节示例和 `mini_d2l.py` 的通用工具。提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

本项目使用 [MIT License](LICENSE)。
