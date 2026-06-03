# 贡献指南

感谢你愿意改进这个学习代码库。本项目优先服务《动手学深度学习》的学习流程，因此贡献时请保持代码清晰、入口直观、默认行为轻量。

## 基本原则

- 保留 `chapter*.py` 的扁平章节结构，除非有明确理由不要做大规模目录重构。
- 默认入口不要启动长时间训练、下载大型数据集或写出提交文件。
- 新增示例应尽量复用 `mini_d2l.py`，不要重新引入第三方 `d2l` 包依赖。
- 能用小 shape 检查验证的内容，优先写成轻量测试或 `inspect_*` 函数。

## 本地检查

建议使用 Python 3.10-3.12：

```bash
pip install -r requirements-dev.txt
python -m compileall -q .
ruff check .
pytest -q
```

如果当前环境没有安装 `torch` 或 `torchvision`，可以先运行不依赖导入深度学习库的静态测试：

```bash
pytest -q tests/test_project_baseline.py
```

## PR 建议

- 一次 PR 聚焦一个主题，例如“补齐测试”“修正文档”“新增某章轻量检查”。
- 在 PR 描述里说明改动原因、验证命令和是否涉及联网数据。
- 不要提交 `data/`、`submission.csv`、模型参数或 notebook checkpoint。
