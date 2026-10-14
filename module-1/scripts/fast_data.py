"""把 CIFAR-10 预加载到内存 + 张量级增强，绕开 PIL/多进程 IPC 瓶颈。
数据划分仍由 scripts.prepare_data.prepare_cifar10_datasets 决定（valid 用它的 subset indices）。
增强等价于 RandomCrop(32, padding=4) + RandomHorizontalFlip。"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_RAW_DIR = os.path.join(".", "data", "cifar-10-batches-py")


def _read_batch(fname):
    with open(os.path.join(_RAW_DIR, fname), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


class CachedCIFAR(Dataset):
    def __init__(self, split="train", indices=None, augment=False):
        if split == "train":
            xs, ys = [], []
            for k in range(1, 6):
                x, y = _read_batch("data_batch_%d" % k)
                xs.append(x)
                ys.append(y)
            self.data = torch.cat(xs)
            self.targets = torch.cat(ys)
        else:
            self.data, self.targets = _read_batch("test_batch")
        if indices is not None:
            idx = torch.as_tensor(list(indices), dtype=torch.long)
            self.data = self.data[idx]
            self.targets = self.targets[idx]
        self.augment = augment

    def __len__(self):
        return self.targets.numel()

    def __getitem__(self, i):
        x = self.data[i]
        if self.augment:
            x = F.pad(x, (4, 4, 4, 4))
            top = int(torch.randint(0, 9, (1,)))
            left = int(torch.randint(0, 9, (1,)))
            x = x[:, top:top + 32, left:left + 32]
            if float(torch.rand(1)) < 0.5:
                x = torch.flip(x, dims=[2])
        x = x.float().div_(255).sub_(0.5).div_(0.5)
        return x, self.targets[i]
