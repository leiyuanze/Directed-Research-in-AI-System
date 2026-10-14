# Module 1: Machine Learning

CIFAR-10 图像分类作业：手写 **SimpleCNN**（< 100K 参数）与 **SimpleViT**（纯 PyTorch 实现，未使用 `nn.Transformer*`），
并完成可选 **SimpleComposition**（bonus）。全部实验使用 WandB 记录，报告使用课程 LaTeX 模板。

## 结果（75 epochs，batch 512，随机种子 0）

| 模型 | 参数量 | 最佳验证精度 | 测试精度（10,000 张） |
| --- | ---: | ---: | ---: |
| SimpleCNN | 85,642 (< 100K) | 79.54% | **79.48%** |
| SimpleViT | 141,386 | 65.33% | 65.01% |
| SimpleComposition (bonus) | 81,194 (< 120K) | 69.97% | **69.98%** |

WandB 项目：https://wandb.ai/2293265560-peking-university/ml-module1-cifar10

## 目录

- `models/` — SimpleCNN / SimpleViT / SimpleComposition
- `scripts/` — `prepare_data.py`（数据划分）、`train.py`、`test.py`；`fast_data.py` 为内存预加载 + 张量增强的加速数据层
- `model_ckpt/` — 各模型 best 权重（`.pth`）与构造超参（`_arg.pickle`）
- `report/` — `report.tex` / `report.pdf`、曲线图（`figures/`）、画图脚本；模板依赖 `config.tex`、`template22.sty`、`reference.bib`
- `runs/` — 训练导出的指标 CSV（报告的曲线数据源）
- `handout/` — 课程原始说明与 `autograder.py`

## 运行

```bash
pip install torch torchvision einops wandb
python scripts/train.py --model SimpleCNN     # 75 epochs（RTX 3050 Ti 上约 13 分钟）
python scripts/test.py  --model SimpleCNN
python handout/autograder.py --q q1           # 精度项自评（需先在项目根目录运行）
```

报告编译（中文姓名需 XeLaTeX + ctex/Fandol 字体）：

```bash
cd report && latexmk -xelatex report.tex
```
