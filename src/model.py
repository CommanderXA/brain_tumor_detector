import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2d = Conv2dBlock(1, 32)
        self.batch_norm = nn.BatchNorm2d(32)
        self.conv2d_1 = Conv2dBlock(32, 64)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.lin = nn.Linear(64*15*15, 128)
        self.lin_1 = nn.Linear(128, 32)
        self.lin_2 = nn.Linear(32, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.conv2d_1(x)
        x = self.batch_norm_2(x)
        x = x.reshape(-1, 64*15*15)
        x = F.relu(self.lin(x))
        x = F.relu(self.lin_1(x))
        x = torch.sigmoid(self.lin_2(x))
        return x

    def parameters_count(self, only_trainable=False) -> int:
        if only_trainable is True:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class Conv2dBlock(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_features, out_features, 3, 2, 0)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(x)
        x = self.pool(x)
        return x
