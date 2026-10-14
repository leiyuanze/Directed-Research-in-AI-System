import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from argparse import ArgumentParser
import pickle

from scripts.prepare_data import prepare_cifar10_datasets
from models.SimpleCNN import SimpleCNN
from models.SimpleViT import SimpleViT
from models.SimpleComposition import SimpleComposition


def count_parameters(model, except_linear=False):
    if except_linear:
        total_params = 0
        for layer in model.children():
            if not isinstance(layer, nn.Linear):
                total_params += sum(p.numel() for p in layer.parameters())
        return total_params
    else:
        return sum(p.numel() for p in model.parameters())


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    model.to(device)
    model_save_path = f"./model_ckpt/cifar10_{model.name()}.pth"
    best_acc = -1.

    # ---- 本地 CSV 记录（画图/报告用，与 WandB 并行）----
    import csv, os
    os.makedirs("./runs", exist_ok=True)
    csv_file = open(f"./runs/cifar10_{model.name()}_metrics.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss_epoch", "val_loss", "val_acc"])
    step_csv_file = open(f"./runs/cifar10_{model.name()}_steps.csv", "w", newline="", encoding="utf-8")
    step_csv_writer = csv.writer(step_csv_file)
    step_csv_writer.writerow(["step", "train_loss_step"])
    
    
    for epoch in range(num_epochs):
        # TODO: Implement the training loop. Remember to call schedulers if they exist; and remeber to record your losses and metrics using WandB.
        # raise NotImplementedError()
        model.train()
        loss_sum, n_seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            n_seen += x.size(0)
            if (n_seen // x.size(0)) % 5 == 0:
                wandb.log({"train_loss_step": loss.item()})
                step_csv_writer.writerow([n_seen // x.size(0), f"{loss.item():.6f}"])

        train_loss = loss_sum / n_seen
        wandb.log({"train_loss_epoch": train_loss, "epoch": epoch + 1})

        # TODO: Implement the Validation phase. Remember to save model state_dict into path `model_save_path` at the best acc; and remeber to record your losses and metrics using WandB.
        # raise NotImplementedError()

        model.eval()
        correct, n_val, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss_sum += criterion(logits, y).item() * x.size(0)
                correct += (logits.argmax(dim = 1) == y).sum().item()
                n_val += x.size(0)

        val_acc = correct / n_val
        val_loss = val_loss_sum / n_val
        wandb.log({"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})
        csv_writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])
        csv_file.flush()

        if scheduler is not None:
            scheduler.step()

        print(f"[{epoch+1:03d}/{num_epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc :
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"    -> saved best ({best_acc*100:.2f}%)")
    csv_file.close()
    step_csv_file.close()
    return best_acc


# Main script
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="SimpleCNN", type=str)
    args = parser.parse_args()

    # TODO: Fix all relevant random seeds for reproducibility.
    # raise NotImplementedError()
    import random, numpy as np
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False   # 4GB 显存：关闭自动调优，避免 OOM

    # TODO: Set your device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: Initialize your WandB settings.
    # raise NotImplementedError()
    wandb.init(project="ml-module1-cifar10", name=f"{args.model}-seed0")
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss_epoch", step_metric="epoch")
    wandb.define_metric("val_acc", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")

    # TODO: Set model hyperparameters, and batch_size,
    #       and initialize `model`, `optimizer`, `scheduler`, 
    #       and `train_transform` (used to initialize train_dataset. You can add data augmentation here. 
    #       You can use all functions in torchvision.transforms. If `train_transform` is set to None, it defaults to the same transform used in the test dataset 
    #       as defined in `prepare_cifar10_datasets`.

    if args.model == "SimpleCNN":
        model_hps = {
            "conv_dim1": 32,
            "conv_dim2": 64,
            "fc_dim": 64,
            "dropout_rate": 0.3,
        }
        batch_size = 512
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        
        model = SimpleCNN(**model_hps)

        total_params = count_parameters(model)
        print(f'Total number of parameters: {total_params}')
        assert total_params < 100_000, "Please make sure your CNN model's total param num is lower than 100K."
        
        optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
        
    elif args.model == "SimpleViT":
        model_hps = {
            "depth": 4,
            "heads": 4,
            "mlp_dim": 128,
            "dim_head": 16,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "image_size": (32, 32),
            "patch_size": (4, 4),
            "channels": 3,
            "dim": 64
        }
        batch_size = 512
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        
        model = SimpleViT(**model_hps)
        
        total_params = count_parameters(model)
        print(f'Total number of parameters: {total_params}')
        
        # TODO: Init optimizer and scheduler (Optionally for scheduler. If you do not use scheduler, set it to None.)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
        
    elif args.model == "SimpleComposition":
        model_hps = {
            "CNN_model_hps": {
                "conv_dim1": 16,
                "conv_dim2": 32,
                "fc_dim": 32,
                "dropout_rate": 0.1,
            },
            "ViT_model_hps": {
                "depth": 2,
                "heads": 4,
                "mlp_dim": 128,
                "dim_head": 16,
                "dropout": 0.1,
                "emb_dropout": 0.1,
                "image_size": (4, 4),
                "patch_size": (2, 2),
                "channels": 32,
                "dim": 64
            }
        }
        batch_size = 512
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        
        model = SimpleComposition(model_hps["CNN_model_hps"], model_hps["ViT_model_hps"])
        
        total_params = count_parameters(model.cnn, except_linear=True) + count_parameters(model.vit, except_linear=False) 
        print(f'Total number of parameters: {total_params}')
        assert total_params < 120_000, "Please make sure your Composition model's total param num is lower than 120K."
        
        # TODO: Init optimizer and scheduler (Optionally for scheduler. If you do not use scheduler, set it to None.)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
        
    else:
        raise ValueError(f"Model {args.model} is not implemented!")
    
    model_ckpt_save_path = f"./model_ckpt/cifar10_{model.name()}_arg.pickle"
    # TODO: Save the `model_hps` to file `model_ckpt_save_path` for testing. Use pickle.
    # raise NotImplementedError()
    import os
    os.makedirs("./model_ckpt", exist_ok=True)
    with open(model_ckpt_save_path, "wb") as f:
        pickle.dump(model_hps, f)

    # Prepare datasets
    train_dataset, valid_dataset, _ = prepare_cifar10_datasets(train_transform)

    # TODO: Initialize corresponding dataloaders. Be careful whether to shuffle in each of the datasets.
    # NOTE: If you're applying more advanced data augmentation in the "transform", consider increasing 'num_workers'
    # to parallelize data processing during training for faster training.
    # 加速：预加载到内存 + 张量级增强（划分沿用上面的 valid_dataset.indices，语义不变）
    from scripts.fast_data import CachedCIFAR
    train_dataset = CachedCIFAR("train", augment=True)
    valid_dataset = CachedCIFAR("test", indices=list(valid_dataset.indices), augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    import time
    start_time = time.time()
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=75)
    
    print(f"Training takes {(time.time() - start_time) / 60} min")
    
    # Finish WandB run
    wandb.finish()
