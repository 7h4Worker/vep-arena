"""TRCA-Net CNN classifier.

Ported from ``trcanet_benchmark_pytorch/trcanet_benchmark/model.py``.
This is the CNN head that operates on TRCA-projected features.
For the TRCA filter computation itself, see ``vep_arena.methods.trcanet``.

Input shape: ``(batch, subbands, class_filters, samples)``.
"""
from __future__ import annotations

import torch
from torch import nn


class TRCANet(nn.Module):

    def __init__(
        self,
        filters: int,
        samples: int,
        subbands: int = 3,
        classes: int = 40,
        dropout: float = 0.1,
        final_dropout: float = 0.95,
    ) -> None:
        super().__init__()
        self.subband_conv = nn.Conv2d(subbands, 1, kernel_size=(1, 1), bias=False)
        self.filter_conv = nn.Conv2d(1, 120, kernel_size=(filters, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.temporal_conv = nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2))
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.refine_conv = nn.Conv2d(120, 120, kernel_size=(1, 10), padding="same")
        self.dropout3 = nn.Dropout(final_dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, subbands, filters, samples)
            flat = self._features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(flat, classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.subband_conv.weight)
        for layer in (self.filter_conv, self.temporal_conv, self.refine_conv, self.classifier):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.subband_conv(x)
        x = self.filter_conv(x)
        x = self.dropout1(x)
        x = self.temporal_conv(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.refine_conv(x)
        x = self.dropout3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x).flatten(1))
