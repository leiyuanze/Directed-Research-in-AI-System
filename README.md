# Directed Research in AI System

课程作业仓库，按模块分目录，各模块互相独立、可单独运行。

| 模块 | 内容 | 目录 |
| --- | --- | --- |
| **Module 0** | Git / GitHub / Hugging Face 实践：冻结 ResNet-50 + 10 类特征原型做 MNIST 推理（82.78%） | [`module-0/`](module-0) |
| **Module 1** | 机器学习：CIFAR-10 上的 SimpleCNN / SimpleViT / SimpleComposition（含报告与 checkpoint） | [`module-1/`](module-1) |

## Module 0

- `module-0/inference.py` — 用 `microsoft/resnet-50` 作冻结特征提取器，构建 0~9 数字特征中心，按余弦相似度分类（82.78%）
- `module-0/readme.md` — 作业总结（Git 分支/提交/reflog、Hugging Face、经验教训）
- `module-0/submission/` — 提交给 Gradescope 的三件套：`submission.bundle`、`git_reflog.txt`、`module0_full_submission.zip`

## Module 1

摘要见 [`module-1/README.md`](module-1/README.md)。核心结果：

| 模型 | 参数量 | 测试精度（10k） |
| --- | ---: | ---: |
| SimpleCNN | 85,642 (<100K) | 79.48% |
| SimpleViT | 141,386 | 65.01% |
| SimpleComposition（bonus） | 81,194 (<120K) | 69.98% |
