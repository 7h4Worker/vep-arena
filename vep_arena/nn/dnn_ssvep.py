"""DNN-SSVEP model (Guney et al. 2022).

Ported from ``dnn_ssvep_pytorch/dnn_ssvep/model.py`` into Arena so that
the external sibling repo is no longer required for inference or training.

Input shape: ``(batch, subbands, channels, samples)``.
"""
from __future__ import annotations

import torch
from torch import nn


class DNNSsvep(nn.Module):

    def __init__(
        self,
        channels: int,
        samples: int,
        subbands: int,
        classes: int = 40,
        dropout: float = 0.1,
        final_dropout: float = 0.95,
    ) -> None:
        super().__init__()
        self.subband_conv = nn.Conv2d(subbands, 1, kernel_size=(1, 1), bias=False)
        self.spatial_conv = nn.Conv2d(1, 120, kernel_size=(channels, 1))
        self.dropout1 = nn.Dropout(dropout)
        self.temporal_conv = nn.Conv2d(120, 120, kernel_size=(1, 2), stride=(1, 2))
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.refine_conv = nn.Conv2d(120, 120, kernel_size=(1, 10), padding="same")
        self.dropout3 = nn.Dropout(final_dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, subbands, channels, samples)
            features = self._features(dummy)
            flat = features.flatten(1).shape[1]
        self.classifier = nn.Linear(flat, classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.subband_conv.weight)
        for layer in (self.spatial_conv, self.temporal_conv, self.refine_conv, self.classifier):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.subband_conv(x)
        x = self.spatial_conv(x)
        x = self.dropout1(x)
        x = self.temporal_conv(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.refine_conv(x)
        x = self.dropout3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = x.flatten(1)
        return self.classifier(x)
