"""Minimal EEGNet (Lawhern et al. 2018).

Ported from ``eegnet_minimal/train_synthetic.py``.

Input shape: ``(batch, 1, channels, samples)``.
"""
from __future__ import annotations

import torch
from torch import nn


class EEGNetMini(nn.Module):

    def __init__(self, channels: int, samples: int, classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=(channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(16, 32, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(0.25),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            features = self.net(dummy)
            flat = features.flatten(1).shape[1]
        self.classifier = nn.Linear(flat, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.classifier(x)
