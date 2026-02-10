import torch
import torch.nn as nn
from torch.distributions import Categorical


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        入力: 特徴量 (Batch, Dim)
        出力: ロジット (Batch, Num_Classes)
        """
        return self.net(x)

    def get_entropy_and_logits(self, x):
        logits = self.forward(x)
        dist = Categorical(logits=logits)  # softmaxに通して確率に変換する。
        entropy = dist.entropy()
        # print("logit = ", logits)
        # print("logit = ", entropy)
        return entropy, logits
        # get_entropy_and_logitsは推論用
