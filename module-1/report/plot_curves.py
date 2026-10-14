"""把训练导出的 CSV（与 WandB 记录的同一组指标）画成报告用的曲线图。
每个 metric 一张图，三个模型画在同一坐标系中，便于对比。"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "./runs"
OUT = "./report/figures"
MODELS = [("SimpleCNN", "SimpleCNN"), ("SimpleViT", "SimpleViT"), ("SimpleComposition", "SimpleComposition")]

METRICS_EPOCH = [
    ("train_loss_epoch", "Training loss (epoch-level)", "train_loss_epoch"),
    ("val_loss", "Validation loss", "val_loss"),
    ("val_acc", "Validation accuracy (%)", "val_acc"),
]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUT, exist_ok=True)
    steps = {n: read_csv(os.path.join(RUNS, "cifar10_%s_steps.csv" % n)) for n, _ in MODELS}
    ep = {n: read_csv(os.path.join(RUNS, "cifar10_%s_metrics.csv" % n)) for n, _ in MODELS}

    # 1) step-level training loss
    plt.figure(figsize=(7, 4))
    for n, lab in MODELS:
        if steps[n]:
            plt.plot([int(r["step"]) for r in steps[n]], [float(r["train_loss_step"]) for r in steps[n]], label=lab, lw=1)
    plt.xlabel("step"); plt.ylabel("training loss"); plt.grid(alpha=.3); plt.legend()
    plt.title("train_loss_step (WandB step-level metric)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_metric_train_loss_step.png"), dpi=150); plt.close()

    # 2) 每个 epoch-level metric 一张图
    for key, title, _ in METRICS_EPOCH:
        plt.figure(figsize=(7, 4))
        for n, lab in MODELS:
            rows = ep[n]
            if rows:
                xs = [int(r["epoch"]) for r in rows]
                ys = [float(r[key]) * (100 if key == "val_acc" else 1) for r in rows]
                plt.plot(xs, ys, label=lab, lw=1.5)
        plt.xlabel("epoch"); plt.ylabel(title); plt.grid(alpha=.3); plt.legend()
        plt.title(title + " (WandB epoch-level metric)")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_metric_%s.png" % key), dpi=150); plt.close()

    for n, _ in MODELS:
        rows = ep[n]
        if rows:
            best = max(float(r["val_acc"]) for r in rows)
            print("%-18s epochs=%d best_val_acc=%.2f%%" % (n, len(rows), best * 100))


if __name__ == "__main__":
    main()
