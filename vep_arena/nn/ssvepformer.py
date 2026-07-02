"""SSVEPFormer model (Chen et al.).

Ported from ``ssvepformer_benchmark_pytorch/ssvepformer_benchmark/model.py``.
Includes single-band ``SSVEPFormerTH`` and filter-bank ``FBSSVEPFormer``.

Input shape for SSVEPFormerTH: ``(batch, channels, samples)``.
Input shape for FBSSVEPFormer: ``(batch, channels, samples)`` — filter bank
applied internally via scipy.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch import flatten, nn


class ChComb(nn.Module):
    def __init__(self, chans: int, samples: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(chans // 2, chans, 1, padding="same")
        self.ln = nn.LayerNorm(samples)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.ln(self.conv(x))))


class Encoder(nn.Module):
    def __init__(self, chans: int, samples: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.channels = chans
        self.ln1 = nn.LayerNorm(samples)
        self.conv = nn.Conv1d(chans, chans, 31, padding="same")
        self.ln2 = nn.LayerNorm(samples)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.ln3 = nn.LayerNorm(samples)
        self.proj = nn.Linear(chans, samples)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut1 = x
        x = self.conv(self.ln1(x))
        x = self.act(self.ln2(x))
        x = self.drop(x) + shortcut1
        shortcut2 = x
        x = self.ln3(x)
        output_channels = []
        for i in range(self.channels):
            channel = self.proj(x[:, :, i])
            output_channels.append(channel.unsqueeze(1))
        x = torch.cat(output_channels, 1)
        x = self.drop2(x) + shortcut2
        return x


class MlpHead(nn.Module):
    def __init__(self, chans: int, samples: int, classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.linear1 = nn.Linear(chans * samples, 6 * classes)
        self.norm = nn.LayerNorm(6 * classes)
        self.activation = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(6 * classes, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = flatten(x, 1)
        x = self.drop(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.drop2(x)
        return self.linear2(x)


class SSVEPFormerTH(nn.Module):
    """Reference-aligned SSVEPFormer for the Tsinghua Benchmark dataset.

    fs=250, 9 channels, 40 classes.  Input: ``(batch, channels, samples)``.
    """

    def __init__(
        self,
        chans: int = 9,
        classes: int = 40,
        fs: int = 250,
        band: tuple[float, float] = (8.0, 64.0),
        resolution: float = 0.25,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.name = "SSVEPFormerTH"
        self.fs = fs
        self.resolution = resolution
        self.nfft = round(fs / resolution)
        self.fft_start = int(round(band[0] / self.resolution))
        self.fft_end = int(round(band[1] / self.resolution)) + 1
        samples = (self.fft_end - self.fft_start) * 2
        filters = 2 * chans
        self.channel_comb = ChComb(filters, samples, drop_rate)
        self.encoder1 = Encoder(filters, samples, drop_rate)
        self.encoder2 = Encoder(filters, samples, drop_rate)
        self.head = MlpHead(filters, samples, classes, drop_rate)
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if hasattr(module, "weight"):
                cls_name = module.__class__.__name__
                if "BatchNorm" not in cls_name and "LayerNorm" not in cls_name:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        x = self.channel_comb(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        return self.head(x)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            samples = x.shape[-1]
            spec = torch.fft.fft(x, n=self.nfft) / samples
            real = spec.real[:, :, self.fft_start : self.fft_end]
            imag = spec.imag[:, :, self.fft_start : self.fft_end]
            return torch.cat((real, imag), axis=-1)


class FBSSVEPFormer(nn.Module):
    """Filter-bank SSVEPFormer.  Input: ``(batch, channels, samples)``."""

    def __init__(
        self,
        fs: int = 250,
        classes: int = 40,
        chans: int = 9,
        subbands: int = 3,
        drop_rate: float = 0.5,
        resolution: float = 0.25,
    ) -> None:
        super().__init__()
        self.name = "FBSSVEPFormer"
        self.fs = fs
        self.bands = [(8.0 * i, 80.0) for i in range(1, subbands + 1)]
        self.subnets = nn.ModuleList(
            [
                SSVEPFormerTH(
                    chans=chans,
                    classes=classes,
                    fs=fs,
                    band=band,
                    resolution=resolution,
                    drop_rate=drop_rate,
                )
                for band in self.bands
            ]
        )
        self.conv = nn.Conv1d(subbands, 1, 1, padding="same")
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for subnet, band in zip(self.subnets, self.bands):
            filtered = self.filter_band(x, band)
            outputs.append(subnet(filtered).unsqueeze(1))
        x = torch.cat(outputs, 1)
        return self.conv(x).squeeze(1)

    def filter_band(self, x: torch.Tensor, band: tuple[float, float]) -> torch.Tensor:
        device = x.device
        with torch.no_grad():
            arr = x.detach().cpu().numpy()
            b, a = butter(4, np.array(band) / (self.fs / 2), btype="bandpass")
            arr = filtfilt(b, a, arr, axis=-1).copy()
        return torch.tensor(arr, dtype=torch.float32, device=device)


SSVEPFormer = SSVEPFormerTH
