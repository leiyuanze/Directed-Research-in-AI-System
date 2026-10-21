# Directed Research in AI System

课程作业仓库，按模块分目录，各模块互相独立、可单独运行。

| 模块 | 内容 | 目录 |
| --- | --- | --- |
| **Module 0** | Git / GitHub / Hugging Face 实践：冻结 ResNet-50 + 10 类特征原型做 MNIST 推理（82.78%） | [`module-0/`](module-0) |
| **Module 1** | 机器学习：CIFAR-10 上的 SimpleCNN / SimpleViT / SimpleComposition（含报告与 checkpoint） | [`module-1/`](module-1) |
| **Module 2** | 计算机视觉：Omnidata 单目深度/法线估计 + 多视角 RGB-D TSDF 重建（含 5 页报告与 Blender 渲染） | [`module-2/`](module-2) |

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

## Module 2

- `module-2/hw.ipynb` — 补全 `read_rgbd_and_pose`，读取 100 视角 RGB-D 并完成 TSDF 重建
- `module-2/task1_infer.py` — Omnidata v2 DPT-Hybrid 深度/法线推理（5 个域）
- `module-2/report/report.pdf` — 5 页报告（五域分析、失败案例、Depth Anything V2 对比、相机约定与 TSDF 超参数）
- `module-2/ML2_2510124204_submission.zip` — Gradescope 提交包

关键结果：相机位姿右乘 `diag(1,-1,-1,1)` 后，重建包围盒 `0.99 x 1.36 x 0.93 m`，与 `recon_tgt.ply` 的顶点集完全一致（最大坐标误差 0.0 mm）。
